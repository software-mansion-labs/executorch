/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(IO_STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}
${define_required_extensions("buffer", SCALE_DTYPE)}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}

$if SCALE_DTYPE == "float":
  #define SCALE_DTYPE_FP32 1

$if IO_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define TILE_N8 ${TILE_N8}

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N8 * 2}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N8 * 8}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int4_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", SCALE_DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}

#include "linear_fp_input_tile_load.glslh"
#include "linear_int4_weight_tile_load.glslh"
#ifndef SCALE_DTYPE_FP32
#include "linear_fp_weight_scales_load.glslh"
#endif
#include "linear_fp_bias_load.glslh"
#include "linear_fp_output_tile_fp_int4_compute.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "linear_fp_output_tile_store.glslh"
// iter137 fp32-linear-acc: use fp32 accumulator across the K-dim matmul.
#include "linear_fp32_acc.glslh"
#ifdef SCALE_DTYPE_FP32
// iter145 fp32-weight-scale: scales kept at fp32 precision through dequant.
#include "linear_fp32_scale_acc.glslh"
#endif

void main() {
  const int out_tile_x = int(gl_GlobalInvocationID.x);
  const int out_tile_y = int(gl_GlobalInvocationID.y);

  const int n = out_tile_x * TILE_N;
  const int m = out_tile_y * TILE_M;

  const int n8 = div_8(n);
  const int n4 = div_4(n);
  const int m4 = div_4(m);

  if (n >= output_sizes.x || m >= output_sizes.y) {
    return;
  }

  const int M = input_sizes.y;
  const int K4 = div_up_4(input_sizes.x);
  const int N4 = div_up_4(output_sizes.x); // number of texels in each row
  const int N8 = div_up_8(output_sizes.x); // number of texels in each row

  // iter137 fp32-linear-acc: fp32 accumulator across K-dim (~1536) matmul.
  LinearFPOutTileFP32 out_tile;
  fp32_initialize(out_tile);

  FPInputTile in_tile;
  Int4WeightTile int4_weight_tile;

#ifdef SCALE_DTYPE_FP32
  FPPerOutChannelScalesFP32 weight_scales_tile;
#else
  FPPerOutChannelParams weight_scales_tile;
#endif
  FPPerOutChannelParams weight_zeros_tile;
  weight_zeros_tile.data[0] = VEC4_T(0.0);
  weight_zeros_tile.data[1] = VEC4_T(0.0);

  const int num_groups = K4 / K4_per_group;

  for (int group_i = 0; group_i < num_groups; ++group_i) {
    // Load quantization scales and zeros for the current group
#ifdef SCALE_DTYPE_FP32
    load_weight_scales_fp32_tile_for_group(
        weight_scales_tile, n4, group_i, N4);
#else
    load_weight_scales_tile_for_group(weight_scales_tile, n4, group_i, N4);
#endif

    for (int k4_inner = 0; k4_inner < K4_per_group; k4_inner++) {
      const int k4 = group_i * K4_per_group + k4_inner;

      load_input_tile_no_checks(in_tile, k4, m, K4, M);
      load_int4_weight_tile(int4_weight_tile, k4, n8, K4);

#ifdef SCALE_DTYPE_FP32
      fp32_scale_fp_accumulate_with_int4_weight(
          out_tile,
          in_tile,
          int4_weight_tile,
          weight_scales_tile,
          weight_zeros_tile);
#else
      fp32_fp_accumulate_with_int4_weight(
          out_tile,
          in_tile,
          int4_weight_tile,
          weight_scales_tile,
          weight_zeros_tile);
#endif
    }
  }

  fp32_write_output_tile_with_checks(out_tile, n4, m, N4, M);
}
