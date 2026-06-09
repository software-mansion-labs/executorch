/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/RepeatInterleave.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Slice.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Softmax.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Transpose.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/runtime/platform/log.h>

namespace vkcompute {

bool is_single_token(ComputeGraph* graph, const ValueRef& q_projected) {
  return graph->size_at<uint32_t>(-3, q_projected) == 1;
}

//
// Resize functions
//

void resize_compute_attn_weights_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights = args.at(0).refs.at(0);
  const ValueRef q_projected = args.at(1).refs.at(0);
  const ValueRef input_pos_symint = resize_args.at(0);

  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  // start_pos may be serialized as either SymInt (dynamic-shape text_decoder
  // prefill/decode) or Int (static-shape vision_encoder with constant 0). The
  // op schema declares SymInt, but recent vulkan_preprocess passes serialize
  // Python literal constants as Int. Dispatch on the actual value type.
  const int32_t input_pos_val = graph->val_is_symint(input_pos_symint)
      ? graph->read_symint(input_pos_symint)
      : graph->extract_scalar<int32_t>(input_pos_symint);

  const uint32_t context_len = seq_len + input_pos_val;

  std::vector<int64_t> out_sizes = {
      1, // batch
      num_q_heads,
      utils::align_up_4(seq_len),
      utils::align_up_4(context_len)};

  graph->virtual_resize(attn_weights, out_sizes);
}

void resize_sdpa_attn_weights_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights_softmax = args.at(0).refs.at(0);
  const ValueRef attn_weights = args.at(1).refs.at(0);

  graph->virtual_resize(attn_weights_softmax, graph->sizes_of(attn_weights));
}

void resize_sdpa_compute_out_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef q_projected = resize_args.at(0);

  graph->virtual_resize(out, graph->sizes_of(q_projected));
}

void resize_sdpa_out(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;

  int arg_idx = 0;
  const ValueRef q_projected = extra_args[arg_idx++];
  const ValueRef out = extra_args[arg_idx++];
  graph->virtual_resize(out, graph->sizes_of(q_projected));
}

//
// Shader dispatch pick functions
//

utils::uvec3 kv_cache_update_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef projected = args.at(1).refs.at(0);

  const uint32_t head_dim_size = graph->size_at<uint32_t>(-1, projected);
  const uint32_t num_heads = graph->size_at<uint32_t>(-2, projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, projected);

  return {utils::div_up_4(head_dim_size), seq_len, num_heads};
}

utils::uvec3 attn_weight_scale_and_mask_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef attn_weight = args.at(0).refs.at(0);

  if (graph->is_buffer_storage(attn_weight)) {
    return {
        graph->size_at<uint32_t>(-1, attn_weight),
        graph->size_at<uint32_t>(-2, attn_weight),
        graph->size_at<uint32_t>(-3, attn_weight),
    };
  } else {
    return graph->logical_limits_of(attn_weight);
  }
}

vkapi::ShaderInfo pick_sdpa_compute_attn_weights_shader_impl(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args,
    const bool force_tiled,
    const bool mask_per_head = false,
    const bool has_softcap = false) {
  const ValueRef q_projected = args.at(1).refs.at(0);
  const ValueRef k_cache = args.at(1).refs.at(1);
  // If a 3rd kRead arg is present, it's the optional attn_mask binding.
  const bool has_mask = args.at(1).refs.size() >= 3;

  const bool is_gemv = !force_tiled && is_single_token(graph, q_projected);

  std::string shader_name = "sdpa_compute_attn_weights";
  if (is_gemv) {
    shader_name += "_coop";
  } else {
    shader_name += "_tiled";
  }

  // Audio extension (Phase 2B): variant suffix is now baked into the yaml's
  // explicit shader_variants NAME field; codegen appends storage and dtype
  // cross-product AFTER it. Order: <base>_<variant>_<storage>_<storage>_<dtype>.
  // This matches the gemma-fork picker's convention so the audio path can
  // route through these (multi-row-correct) shaders.
  if (has_mask) {
    if (mask_per_head) {
      shader_name += has_softcap ? "_mask_phmask_softcap" : "_mask_phmask";
    } else {
      shader_name += has_softcap ? "_mask_softcap" : "_mask";
    }
  } else {
    shader_name += "_nomask";
  }
  add_storage_type_suffix(shader_name, graph->storage_type_of(q_projected));
  add_storage_type_suffix(shader_name, graph->storage_type_of(k_cache));
  add_dtype_suffix(shader_name, graph->dtype_of(q_projected));

  return VK_KERNEL_FROM_STR(shader_name);
}

// Picker for the fused attn-weights + softmax shader (decode S=1 only).
vkapi::ShaderInfo pick_sdpa_compute_attn_weights_coop_softmax_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef q_projected = args.at(1).refs.at(0);
  const ValueRef k_cache = args.at(1).refs.at(1);
  // If a 3rd kRead arg is present, it's the optional attn_mask binding.
  const bool has_mask = args.at(1).refs.size() >= 3;

  std::string shader_name = "sdpa_compute_attn_weights_coop_softmax";

  add_storage_type_suffix(shader_name, graph->storage_type_of(q_projected));
  add_storage_type_suffix(shader_name, graph->storage_type_of(k_cache));
  shader_name += has_mask ? "_mask" : "_nomask";
  add_dtype_suffix(shader_name, graph->dtype_of(q_projected));

  return VK_KERNEL_FROM_STR(shader_name);
}

vkapi::ShaderInfo pick_sdpa_compute_attn_weights_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return pick_sdpa_compute_attn_weights_shader_impl(
      graph, args, resize_args, /*force_tiled=*/false);
}

utils::uvec3 pick_sdpa_compute_attn_weights_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef q_projected = args.at(1).refs.at(0);
  const ValueRef k_cache = args.at(1).refs.at(1);

  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  // CMD-REUSE optimization: dispatch over the MAX possible context_len
  // (a graph-time constant equal to k_cache.size(-3)) instead of the
  // current context_len (= seq_len + input_pos, which grows by 1 per
  // decode token). Keeping the global WG size constant at decode lets
  // the deferred-command-buffer cache stay live (otherwise every token
  // would trigger requires_reencode_). Out-of-range threads early-return
  // at the top of the shader (bounds check `if (c >= context_len ...)`).
  const uint32_t max_context_len = graph->size_at<uint32_t>(-3, k_cache);

  const uint32_t N4 = utils::div_up_4(max_context_len);
  const uint32_t M4 = utils::div_up_4(seq_len);

  return {N4, M4, num_q_heads};
}

utils::uvec3 pick_sdpa_compute_attn_weights_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {1, 64, 1};
  } else {
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
}

utils::uvec3 pick_sdpa_attn_weights_softmax_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef q_projected = resize_args.at(0);

  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  return {1, seq_len, num_q_heads};
}

utils::uvec3 pick_sdpa_attn_weights_softmax_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return {64, 1, 1};
}

// Fused attn-weights compute + softmax dispatch geometry. Same shape as the
// standalone softmax shader (one WG per (s, q_h), 64 workers per WG).
utils::uvec3 pick_sdpa_compute_attn_weights_coop_softmax_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef q_projected = args.at(1).refs.at(0);

  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  return {1, seq_len, num_q_heads};
}

utils::uvec3 pick_sdpa_compute_attn_weights_coop_softmax_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  return {64, 1, 1};
}

vkapi::ShaderInfo pick_sdpa_compute_out_shader_impl(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args,
    const bool force_tiled) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef v_cache = args.at(1).refs.at(1);

  const ValueRef q_projected = resize_args.at(0);

  const bool is_gemv = !force_tiled && is_single_token(graph, q_projected);

  std::string shader_name = "sdpa_compute_out";
  if (is_gemv) {
    shader_name += "_coop";
  } else {
    shader_name += "_tiled";
  }

  add_storage_type_suffix(shader_name, graph->storage_type_of(out));
  add_storage_type_suffix(shader_name, graph->storage_type_of(v_cache));
  add_dtype_suffix(shader_name, graph->dtype_of(out));

  return VK_KERNEL_FROM_STR(shader_name);
}

vkapi::ShaderInfo pick_sdpa_compute_out_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return pick_sdpa_compute_out_shader_impl(
      graph, args, resize_args, /*force_tiled=*/false);
}

utils::uvec3 pick_sdpa_compute_out_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef q_projected = resize_args.at(0);

  const uint32_t head_dim = graph->size_at<uint32_t>(-1, q_projected);
  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  const uint32_t N4 = utils::div_up_4(head_dim);
  const uint32_t M4 = utils::div_up_4(seq_len);

  return {N4, M4, num_q_heads};
}

utils::uvec3 pick_sdpa_compute_out_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {1, 64, 1};
  } else {
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
}

//
// Dispatch nodes
//

void add_sdpa_kv_cache_update_node(
    ComputeGraph& graph,
    const ValueRef input_pos_symint,
    const ValueRef projected,
    const ValueRef cache) {
  std::string kernel_name("sdpa_kv_cache_update");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(cache));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(projected));
  add_dtype_suffix(kernel_name, graph.dtype_of(projected));

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(cache),
      graph.sizes_ubo(projected),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      kv_cache_update_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{cache, vkapi::kWrite}, {projected, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {input_pos_symint},
      // Resizing Logic
      nullptr));
}

void add_sdpa_compute_attn_weights_node(
    ComputeGraph& graph,
    const ValueRef q_projected,
    const ValueRef k_cache,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights,
    const float inv_scale_override = -1.0f,
    const ValueRef attn_mask = kDummyValueRef,
    const bool force_tiled = false,
    const bool mask_per_head = false,
    const float softcap_value = 0.0f) {
  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = (inv_scale_override >= 0.0f)
      ? inv_scale_override
      : 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  const bool has_mask =
      (attn_mask != kDummyValueRef) && !graph.val_is_none(attn_mask);
  const bool has_softcap = softcap_value > 0.0f;

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q_projected),
      graph.sizes_ubo(k_cache),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  std::vector<ArgGroup> arg_groups;
  arg_groups.push_back({attn_weights, vkapi::kWrite});
  if (has_mask) {
    arg_groups.push_back({{q_projected, k_cache, attn_mask}, vkapi::kRead});
  } else {
    arg_groups.push_back({{q_projected, k_cache}, vkapi::kRead});
  }

  // Stage-3-W workaround: gemma's decode path triggers an apparent compiler
  // bug in sdpa_compute_attn_weights_coop SPIR-V where odd q_h values produce
  // garbage at S=1 + start_pos>0 with GQA broadcast. Caller may set
  // force_tiled=true to dispatch the tiled shader at S=1 instead. Tiled is
  // sub-optimal for gemv-shaped matmul but is correct (TILE_M=4 with bounds
  // checks covers any S<=4). Stage-1+2 prefill validation (cos=1.0) confirms
  // tiled works with HAS_MASK, so this is a safe fallback.
  //
  // Audio extension (Phase 2B): mask_per_head / softcap flags reroute the
  // audio_encoder path off the buggy gemma compute-attn-weights SPIR-V (which
  // has a multi-row corruption bug at row 1+) and onto these multi-row-correct
  // Llama shaders. The picker selects the `_mask_phmask[_softcap]` variant.
  std::function<vkapi::ShaderInfo(
      ComputeGraph*,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&)> pick_shader_fn;
  if (force_tiled || mask_per_head || has_softcap) {
    pick_shader_fn = [force_tiled, mask_per_head, has_softcap](
        ComputeGraph* g,
        const std::vector<ArgGroup>& a,
        const std::vector<ValueRef>& r) -> vkapi::ShaderInfo {
      return pick_sdpa_compute_attn_weights_shader_impl(
          g, a, r, force_tiled, mask_per_head, has_softcap);
    };
  } else {
    pick_shader_fn = pick_sdpa_compute_attn_weights_shader;
  }

  // Push constants for HAS_SOFTCAP variant: pair of floats (softcap, inv_softcap).
  // Inactive variants ignore them (block is not declared when HAS_SOFTCAP=0).
  const float inv_softcap_value = has_softcap ? (1.0f / softcap_value) : 0.0f;
  std::array<float, 2> pcs = {softcap_value, inv_softcap_value};
  std::vector<PushConstantDataInfo> push_constants;
  if (has_softcap) {
    push_constants.push_back(
        PushConstantDataInfo(&pcs[0], sizeof(float) * 2));
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_shader_fn,
      pick_sdpa_compute_attn_weights_global_wg_size,
      pick_sdpa_compute_attn_weights_local_wg_size,
      // Inputs and Outputs
      arg_groups,
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {scale_val},
      // Resize Args
      {input_pos_symint},
      // Resizing Logic
      resize_compute_attn_weights_node));
}

// Decode-only fused attn-weights compute + softmax. Writes softmax-normalized
// attn_weights directly to `attn_weights_softmax` (the buffer compute_out
// reads) so the standalone softmax dispatch can be skipped at decode.
//
// Geometry differs from add_sdpa_compute_attn_weights_node: one WG per
// (s, q_h) with 64 workers iterating ALL c4 cooperatively (vs. one WG per
// (c4, q_h) in the unfused variant). This is required because the softmax
// reduces across the c axis; intra-WG shared-mem reduction needs the entire
// row inside the same workgroup.
void add_sdpa_compute_attn_weights_with_softmax_node(
    ComputeGraph& graph,
    const ValueRef q_projected,
    const ValueRef k_cache,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax,
    const float inv_scale_override = -1.0f,
    const ValueRef attn_mask = kDummyValueRef) {
  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = (inv_scale_override >= 0.0f)
      ? inv_scale_override
      : 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  const bool has_mask =
      (attn_mask != kDummyValueRef) && !graph.val_is_none(attn_mask);

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q_projected),
      graph.sizes_ubo(k_cache),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  std::vector<ArgGroup> arg_groups;
  arg_groups.push_back({attn_weights_softmax, vkapi::kWrite});
  if (has_mask) {
    arg_groups.push_back({{q_projected, k_cache, attn_mask}, vkapi::kRead});
  } else {
    arg_groups.push_back({{q_projected, k_cache}, vkapi::kRead});
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_compute_attn_weights_coop_softmax_shader,
      pick_sdpa_compute_attn_weights_coop_softmax_global_wg_size,
      pick_sdpa_compute_attn_weights_coop_softmax_local_wg_size,
      // Inputs and Outputs
      arg_groups,
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {scale_val},
      // Resize Args
      {input_pos_symint},
      // Resizing Logic
      resize_compute_attn_weights_node));
}

void add_sdpa_attn_weights_softmax_node(
    ComputeGraph& graph,
    const ValueRef attn_weights,
    const ValueRef q_projected,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax) {
  std::string shader_name = "sdpa_attn_weights_softmax";
  add_storage_type_suffix(
      shader_name, graph.storage_type_of(attn_weights_softmax));
  add_dtype_suffix(shader_name, graph.dtype_of(attn_weights_softmax));

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q_projected),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name),
      pick_sdpa_attn_weights_softmax_global_wg_size,
      pick_sdpa_attn_weights_softmax_local_wg_size,
      // Inputs and Outputs
      {{attn_weights_softmax, vkapi::kWrite}, {attn_weights, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {q_projected, input_pos_symint},
      // Resizing Logic
      resize_sdpa_attn_weights_softmax_node));
}

void add_sdpa_compute_out_node(
    ComputeGraph& graph,
    const ValueRef attn_weights_softmax,
    const ValueRef v_cache,
    const ValueRef q_projected,
    const ValueRef input_pos_symint,
    const ValueRef out,
    const bool force_tiled = false) {
  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q_projected),
      graph.sizes_ubo(v_cache),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  // Stage-3-W workaround: force the tiled shader at S=1 to bypass the
  // sdpa_compute_out_coop SPIR-V bug where odd q_h values produce garbage
  // at decode (S=1 + start_pos>0) with GQA broadcast. Tiled at S=1 is
  // sub-optimal but correct.
  std::function<vkapi::ShaderInfo(
      ComputeGraph*,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&)> pick_shader_fn;
  if (force_tiled) {
    pick_shader_fn = [](
        ComputeGraph* g,
        const std::vector<ArgGroup>& a,
        const std::vector<ValueRef>& r) -> vkapi::ShaderInfo {
      return pick_sdpa_compute_out_shader_impl(
          g, a, r, /*force_tiled=*/true);
    };
  } else {
    pick_shader_fn = pick_sdpa_compute_out_shader;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_shader_fn,
      pick_sdpa_compute_out_global_wg_size,
      pick_sdpa_compute_out_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{attn_weights_softmax, v_cache}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {q_projected, input_pos_symint},
      // Resizing Logic
      resize_sdpa_compute_out_node));
}

//
// High level operator impl
//

void update_cache_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef value = args[arg_idx++];
  const ValueRef cache = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef out = args[arg_idx++];

  // Unused variables
  (void)out;

  VK_CHECK_COND(graph.size_at<int32_t>(-4, value) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, cache) == 1);
  VK_CHECK_COND(
      graph.size_at<int32_t>(-1, value) == graph.size_at<int32_t>(-1, cache));
  VK_CHECK_COND(
      graph.size_at<int32_t>(-2, value) == graph.size_at<int32_t>(-2, cache));

  add_sdpa_kv_cache_update_node(graph, input_pos_symint, value, cache);
}

void sdpa_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_cache = args[arg_idx++];
  const ValueRef v_cache = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef dropout_p = args[arg_idx++];
  const ValueRef is_causal = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];

  // Output tensors
  const ValueRef out = args[arg_idx++];

  // Batches must be 1
  VK_CHECK_COND(graph.size_at<int32_t>(-4, q_projected) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, k_cache) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, v_cache) == 1);
  // k and v projected must have the same shape
  VK_CHECK_COND(graph.sizes_of(k_cache) == graph.sizes_of(v_cache));
  // head dim must match between tensors
  VK_CHECK_COND(
      graph.size_at<int32_t>(-1, q_projected) ==
      graph.size_at<int32_t>(-1, k_cache));
  // All tensors must have the packed dim be the width (head) dimension
  VK_CHECK_COND(graph.packed_dim_of(q_projected) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(k_cache) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(v_cache) == WHCN::kWidthDim);
  // Some variables are not supported yet
  VK_CHECK_COND(
      graph.val_is_none(dropout_p) ||
      graph.extract_scalar<double>(dropout_p) == 0);
  VK_CHECK_COND(graph.val_is_none(scale));
  // is_causal is assumed to be true in the current implementation.
  VK_CHECK_COND(
      graph.val_is_none(is_causal) || graph.extract_scalar<bool>(is_causal));
  VK_CHECK_COND(graph.val_is_none(attn_mask));

  const int64_t num_q_heads = graph.size_at<int64_t>(-2, q_projected);
  int64_t max_seq_len = graph.size_at<int64_t>(-3, q_projected);
  const int64_t max_context_len = graph.size_at<int32_t>(-3, k_cache);

  const utils::StorageType attn_weights_storage =
      graph.storage_type_of(q_projected);

  // If using buffer storage for attn weights, we need to ensure that the buffer
  // numel limit is not exceeded. If needed, manually adjust max_seq_len based
  // on the buffer numel limit.
  if (attn_weights_storage == utils::kBuffer) {
    const int64_t max_buffer_numel = graph.max_buffer_numel();
    if (num_q_heads * max_seq_len * max_context_len >= max_buffer_numel) {
      // Compute the maximum possible value for max_seq_len that will hit
      // the buffer numel limit.
      max_seq_len = max_buffer_numel / (num_q_heads * max_context_len);
      // Adjust down to the nearest multiple of 4 to make sure the limit is
      // not hit.
      if (max_seq_len % 4 != 0) {
        max_seq_len = (max_seq_len / 4) * 4;
      } else {
        max_seq_len -= 4;
      }
    }
  }

  std::vector<int64_t> attn_weight_full_sizes = {
      1, // batch
      num_q_heads,
      max_seq_len,
      max_context_len};

  TmpTensor attn_weights(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  TmpTensor attn_weights_softmax(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  // At decode (S=1), fuse the matmul + softmax into a single dispatch.
  // The fused shader writes softmax-normalized values directly to
  // attn_weights_softmax (the buffer compute_out reads), eliminating the
  // separate softmax dispatch and the round-trip on attn_weights.
  const bool fuse_softmax = is_single_token(&graph, q_projected);

  if (fuse_softmax) {
    add_sdpa_compute_attn_weights_with_softmax_node(
        graph,
        q_projected,
        k_cache,
        input_pos_symint,
        attn_weights_softmax);
  } else {
    add_sdpa_compute_attn_weights_node(
        graph, q_projected, k_cache, input_pos_symint, attn_weights);

    add_sdpa_attn_weights_softmax_node(
        graph,
        attn_weights,
        q_projected,
        input_pos_symint,
        attn_weights_softmax);
  }

  add_sdpa_compute_out_node(
      graph, attn_weights_softmax, v_cache, q_projected, input_pos_symint, out);
}

void sdpa_with_kv_cache_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_projected = args[arg_idx++];
  const ValueRef v_projected = args[arg_idx++];
  const ValueRef k_cache_data = args[arg_idx++];
  const ValueRef v_cache_data = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef sequence_len = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef dropout_p = args[arg_idx++];
  const ValueRef is_causal = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];

  // Output tensors
  const ValueRef out = args[arg_idx++];

  (void)sequence_len;

  utils::StorageType cache_storage = graph.storage_type_of(q_projected);
  const ValueRef k_cache =
      graph.add_tensor_like(k_cache_data, cache_storage, utils::kWidthPacked);
  const ValueRef v_cache =
      graph.add_tensor_like(v_cache_data, cache_storage, utils::kWidthPacked);

  update_cache_impl(graph, {k_projected, k_cache, input_pos_symint, -1});
  update_cache_impl(graph, {v_projected, v_cache, input_pos_symint, -1});

  sdpa_impl(
      graph,
      {q_projected,
       k_cache,
       v_cache,
       input_pos_symint,
       attn_mask,
       dropout_p,
       is_causal,
       scale,
       out});
}

void compute_attn_weight_with_kv_cache_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_projected = args[arg_idx++];
  const ValueRef v_projected = args[arg_idx++];
  const ValueRef k_cache_data = args[arg_idx++];
  const ValueRef v_cache_data = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef sequence_len = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  (void)attn_mask;
  const ValueRef dropout_p = args[arg_idx++];
  (void)dropout_p;
  const ValueRef is_causal = args[arg_idx++];
  (void)is_causal;
  const ValueRef scale = args[arg_idx++];
  (void)scale;

  // Output tensors
  const ValueRef out = args[arg_idx++];

  (void)sequence_len;

  const utils::StorageType cache_storage = graph.storage_type_of(q_projected);
  const ValueRef k_cache =
      graph.add_tensor_like(k_cache_data, cache_storage, utils::kWidthPacked);
  const ValueRef v_cache =
      graph.add_tensor_like(v_cache_data, cache_storage, utils::kWidthPacked);

  update_cache_impl(graph, {k_projected, k_cache, input_pos_symint, -1});
  update_cache_impl(graph, {v_projected, v_cache, input_pos_symint, -1});

  add_sdpa_compute_attn_weights_node(
      graph, q_projected, k_cache, input_pos_symint, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
  VK_REGISTER_OP(update_cache.default, update_cache_impl);
  VK_REGISTER_OP(llama.custom_sdpa.default, sdpa_impl);
  VK_REGISTER_OP(
      testing.compute_attn_weight_with_kv_cache.default,
      compute_attn_weight_with_kv_cache_impl);
}

} // namespace vkcompute
