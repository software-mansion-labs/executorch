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

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

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

// Gemma4 exports Q/K in DHSB layout ([B, S, H, D]); the tile-load helpers
// default to DSHB, which would read transposed offsets and corrupt attention.
#define Q_LAYOUT DHSB
#define K_LAYOUT DHSB
#include "sdpa_fp_q_projected_tile_load.glslh"
#include "sdpa_fp_k_cache_tile_load.glslh"
// iter136 fp32-attn-acc: load fp16 storage but accumulate the QK matmul
// in fp32. linear_fp_output_tile_fp_compute.glslh is still pulled in by
// downstream linear shaders we don't touch.
#include "gemma_sdpa_fp32_attn_weight_tile_store.glslh"

/*
 * Gemma4-flavored attention-weight compute (tiled variant).
 *
 * Differences vs sdpa_compute_attn_weights_tiled:
 *   - scale = 1.0 (Gemma4 absorbs 1/sqrt(d) into QKV-norm), so the
 *     inv_scale spec const + multiply are gone.
 *   - When HAS_MASK == 1 the caller provides an additive mask of shape
 *     (1, 1, S_q, C); the tile is broadcast-added rather than relying
 *     on the causal-only -inf masking.
 *   - The "tile entirely in causal-mask region" short-circuit is gated
 *     to the no-mask path; with a caller mask the mask itself encodes
 *     causality (sliding-window mask is non-trivial).
 */

void main() {
  const int tile_idx_x = int(gl_GlobalInvocationID.x);
  const int tile_idx_y = int(gl_GlobalInvocationID.y);
  // idx along output num_q_heads dim
  const int q_h = int(gl_GlobalInvocationID.z);

  // idx along the output context_len dim
  const int c = tile_idx_x * TILE_N;
  const int c4 = div_4(c);

  // idx along the output seq_len dim
  const int s = tile_idx_y * TILE_M;
  const int s4 = div_4(s);

  // head dim and its texel size, over which the dot product is accumulated
  const int D = q_projected_sizes.x;
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

  const int context_len = input_pos + S;
  const int context_texel_len = div_up_4(context_len);

  // bounds check
  if (c >= context_len || s >= S || q_h >= Q_H) {
    return;
  }

  // iter136: fp32 accumulator across D4 dot product (head_dim ~ 512).
  SDPAFPOutTileFP32 out_tile;
  fp32_initialize(out_tile);

  FPInputTile q_tile;
  FPWeightTile w_tile;

#ifdef HAS_MASK
  // With a caller mask we always run the matmul (mask encodes causality).
  for (int d4 = 0; d4 < D4; d4++) {
    load_q_projected_tile_with_checks(
      q_tile, d4, s, q_h, D4, D, S, Q_H);
    load_k_cache_tile_with_checks(
      w_tile, d4, c, kv_h, D4, D, context_len, C, KV_H);
    fp32_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
  }
  fp32_apply_mask_only_with_mask(out_tile, input_pos, c, s, q_h, S, C4);
#ifdef HAS_SOFTCAP
  // Audio extension (Phase 2B): tanh(x/cap) * cap, AFTER mask add.
  fp32_apply_softcap_tile(out_tile, softcap, inv_softcap);
#endif
#else
  // No-mask path: keep the "tile entirely in causal mask region"
  // short-circuit.
  bool tile_in_mask_region = c > (input_pos + s + (TILE_M - 1));
  if (tile_in_mask_region) {
    fp32_set_out_tile_to_vec(out_tile, vec4(fp32_negative_infinity_val));
  } else {
    for (int d4 = 0; d4 < D4; d4++) {
      load_q_projected_tile_with_checks(
      q_tile, d4, s, q_h, D4, D, S, Q_H);
      load_k_cache_tile_with_checks(
      w_tile, d4, c, kv_h, D4, D, context_len, C, KV_H);
      fp32_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
    }
    fp32_apply_mask_only_causal(out_tile, input_pos, c, s);
  }
#endif

  fp32_store_attn_weight_tile_with_checks(
    out_tile, c4, s, q_h, context_texel_len, S_aligned, Q_H);
}
