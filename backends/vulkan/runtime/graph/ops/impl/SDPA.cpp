/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <memory>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/SDPA.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/RepeatInterleave.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Slice.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/matmul/MatMul.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <cmath>

namespace vkcompute {

namespace {

// A fixed local work group, never refitted to the global grid.
LocalWorkGroup fixed_lwg(const utils::uvec3& size) {
  return LocalWorkGroup(size, size[0u] * size[1u] * size[2u]);
}

} // namespace

namespace {

//
// Common dimension helper: folds the axis-swap for LLM vs fused Q layouts.
// `input_pos_symint` is used only for LLM (context_len = S + input_pos);
// pass kDummyValueRef for FUSED.
//
struct SDPADims {
  int64_t B = 1;
  int64_t H = 0;
  int64_t S = 0;
  int64_t D = 0;
  int64_t context_len = 0; // LLM: S + input_pos_val; FUSED: size_at(-2, k)
  int64_t max_context_len = 0; // LLM: size_at(-3, k); FUSED: size_at(-2, k)
};

SDPADims compute_sdpa_dims(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const SDPAMode mode) {
  SDPADims d;
  d.D = graph.size_at<int64_t>(-1, q);
  if (mode == SDPAMode::LLM) {
    // Q: [B=1, S, H, D] (DHSB), K: [B=1, C_max, H_kv, D]
    // `k` may be kDummyValueRef in dispatch pickers that don't need it;
    // max_context_len is only read when k is valid.
    d.B = 1;
    d.H = graph.size_at<int64_t>(-2, q);
    d.S = graph.size_at<int64_t>(-3, q);
    d.max_context_len = is_valid(k) ? graph.size_at<int64_t>(-3, k) : 0;
    const int32_t input_pos_val =
        is_valid(input_pos_symint) ? graph.read_symint(input_pos_symint) : 0;
    d.context_len = d.S + input_pos_val;
  } else {
    // Q: [B, H, S, D] (DSHB), K: [B, H_kv, L, D]
    d.B = graph.size_at<int64_t>(-4, q);
    d.H = graph.size_at<int64_t>(-3, q);
    d.S = graph.size_at<int64_t>(-2, q);
    d.context_len = graph.size_at<int64_t>(-2, k);
    d.max_context_len = d.context_len;
  }
  return d;
}

} // namespace

bool is_single_token(ComputeGraph* graph, const ValueRef& q_projected) {
  return graph->size_at<uint32_t>(-3, q_projected) == 1;
}

// Context length past which the separate attn-weights and softmax pair is
// faster than the fused dispatch. The fused shader reduces the whole row
// inside one work group, so its parallelism is capped at num_q_heads work
// groups; the separate pair spreads the same work over one work group per
// context texel. Below the cutoff the saved dispatch and the saved round trip
// of attn_weights through global memory dominate; above it the lost
// parallelism does. Measured on Adreno 840 only (Qwen3-0.6B, crossover near
// 220 tokens); kShaderOverrideForceSeparateQkSoftmax A/Bs the two paths on one
// shape.
//
// The fused shader also sizes its shared attn_weights row to this cutoff: its
// MAX_CONTEXT_TEXEL_LEN parameter in sdpa_compute_attn_weights_coop.yaml must
// equal kFusedSoftmaxMaxContextTexels, and the shader writes nothing for a
// longer context.
constexpr int64_t kFusedSoftmaxMaxContextTexels = 64;
constexpr int64_t kFusedSoftmaxMaxContextLen =
    kFusedSoftmaxMaxContextTexels * 4;

// The fused shader's shared memory: four row arrays (one per texel component)
// plus the softmax helper's two 64-entry reductions, all fp32 in the worst
// case. Keep it within the 16 KiB maxComputeSharedMemorySize that Vulkan
// guarantees, so no device rejects the pipeline.
static_assert(
    (4 * kFusedSoftmaxMaxContextTexels + 2 * 64) * sizeof(float) <= 16384,
    "fused QK+softmax shared memory exceeds the Vulkan minimum limit");

//
// Resize functions
//

// Unified attn_weights resize. In LLM mode the shape is padded to multiples of
// 4 in the S/context_len dims (to match the tiled shader's iteration space);
// in fused mode it's the unpadded [B, H, S, L].
// resize_args layout: [q, k, input_pos_symint_or_dummy, mode_as_int]
void resize_sdpa_attn_weights_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights = args.at(0).refs.at(0);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPAMode mode = static_cast<SDPAMode>(resize_args.at(3));

  std::vector<int64_t> out_sizes;
  if (mode == SDPAMode::LLM) {
    const int64_t num_q_heads = graph->size_at<int64_t>(-2, q);
    const int64_t seq_len = graph->size_at<int64_t>(-3, q);
    const int32_t input_pos_val = graph->read_symint(input_pos_symint);
    const int64_t context_len = seq_len + input_pos_val;
    out_sizes = {
        1,
        num_q_heads,
        static_cast<int64_t>(utils::align_up_4(seq_len)),
        static_cast<int64_t>(utils::align_up_4(context_len))};
  } else {
    const int64_t B = graph->size_at<int64_t>(-4, q);
    const int64_t H = graph->size_at<int64_t>(-3, q);
    const int64_t S = graph->size_at<int64_t>(-2, q);
    const int64_t L = graph->size_at<int64_t>(-2, k);
    out_sizes = {B, H, S, L};
  }
  graph->virtual_resize(attn_weights, out_sizes);
}

// Softmax preserves attn_weights shape exactly; identical across modes.
void resize_sdpa_attn_weights_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights_softmax = args.at(0).refs.at(0);
  const ValueRef attn_weights = args.at(1).refs.at(0);

  graph->virtual_resize(attn_weights_softmax, graph->sizes_of(attn_weights));
}

// Out matches Q's shape in both modes. resize_args[0] = q.
void resize_sdpa_out_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef q = resize_args.at(0);

  graph->virtual_resize(out, graph->sizes_of(q));
}

//
// Shader dispatch pick functions
//

GlobalWorkGrid kv_cache_update_gwg(
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

  return GlobalWorkGrid(
      {utils::div_up_4(head_dim_size), seq_len, num_heads}, kTiledWorkGrid);
}

// resize_args layout for SDPA dispatch pickers mirrors the node creation
// helper: [q, k, input_pos_symint_or_dummy, mode_as_int].
static inline SDPAMode mode_of(const std::vector<ValueRef>& resize_args) {
  return static_cast<SDPAMode>(resize_args.at(3));
}

// Upper bound on the GQA group size the coop-GQA shader supports; must match
// MAX_GROUP_SIZE in the GQA branch of sdpa_compute_out_coop.glsl.
constexpr int64_t kMaxGqaGroupSize = 8;

// Whether the LLM decode AV path should use the GQA-reuse coop shader: one
// workgroup computes all G = Hq / Hkv query heads sharing a KV head, loading
// each V texel once. Requires GQA (Hq > Hkv, evenly divisible) and a group size
// within the shader's compile-time bound.
bool use_gqa_av_coop(
    ComputeGraph* graph,
    const int64_t num_q_heads,
    const int64_t num_kv_heads) {
  (void)graph;
  if (num_kv_heads <= 0 || num_q_heads <= num_kv_heads) {
    return false;
  }
  if (num_q_heads % num_kv_heads != 0) {
    return false;
  }
  const int64_t group_size = num_q_heads / num_kv_heads;
  return group_size <= kMaxGqaGroupSize;
}

// Resolve whether the AV decode path uses the GQA-reuse coop shader, honoring
// the test-only shader_override knob (see SDPA.h): auto selects via
// use_gqa_av_coop; kShaderOverrideForceNonGqa forces off; any other value
// except kShaderOverrideForceSeparateQkSoftmax, which only concerns the QK and
// softmax nodes, forces the GQA family on.
bool resolve_use_gqa(
    ComputeGraph* graph,
    const ValueRef shader_override,
    const int64_t num_q_heads,
    const int64_t num_kv_heads) {
  if (shader_override == kDummyValueRef ||
      graph->extract_scalar<int64_t>(shader_override) ==
          kShaderOverrideForceSeparateQkSoftmax) {
    return use_gqa_av_coop(graph, num_q_heads, num_kv_heads);
  }
  const bool force_gqa = graph->extract_scalar<int64_t>(shader_override) !=
      kShaderOverrideForceNonGqa;
  // Forcing the GQA shader on an ineligible shape would silently drop query
  // heads (z-dispatch = Hkv cannot cover Hq, or group size exceeds the shader's
  // fixed accumulator array). Fail loudly instead of producing garbage output.
  if (force_gqa) {
    VK_CHECK_COND(
        num_kv_heads > 0 && num_q_heads % num_kv_heads == 0 &&
            num_q_heads / num_kv_heads <= kMaxGqaGroupSize,
        "forcing GQA via shader_override requires a GQA-eligible shape: Hq "
        "divisible by Hkv and group size <= kMaxGqaGroupSize");
  }
  return force_gqa;
}

// Resolve whether the GQA-reuse AV path uses the head_dim output-tiled (_tile2)
// variant. Auto (no override) and kShaderOverrideForceGqa defer to the vendor
// gate (tile2 on Adreno); the force values pin the variant on any device (see
// SDPA.h), giving the Adreno-only tile2 variant deterministic test coverage.
bool resolve_use_tile2(ComputeGraph* graph, const ValueRef shader_override) {
  if (shader_override != kDummyValueRef) {
    const int64_t ov = graph->extract_scalar<int64_t>(shader_override);
    if (ov == kShaderOverrideForceTile2) {
      return true;
    }
    if (ov == kShaderOverrideForceBase) {
      return false;
    }
  }
  return graph->device_is_adreno();
}

vkapi::ShaderInfo pick_sdpa_qk_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    const ValueRef q_projected = args.at(1).refs.at(0);
    const ValueRef k_cache = args.at(1).refs.at(1);
    const bool is_gemv = is_single_token(graph, q_projected);
    // An additive attn_mask is bound as a third input: q, k, attn_mask.
    const bool has_mask = args.at(1).refs.size() >= 3;

    std::string shader_name = "sdpa_compute_attn_weights";
    shader_name += is_gemv ? "_coop" : "_tiled";
    if (has_mask) {
      shader_name += "_mask";
    }
    add_storage_type_suffix(shader_name, graph->storage_type_of(q_projected));
    add_storage_type_suffix(shader_name, graph->storage_type_of(k_cache));
    add_dtype_suffix(shader_name, graph->dtype_of(q_projected));
    return VK_KERNEL_FROM_STR(shader_name);
  } else {
    const ValueRef q = args.at(1).refs.at(0);
    const ValueRef k = args.at(1).refs.at(1);
    // Fused path uses bias variant iff attn_mask was provided (signalled via
    // 3 inputs in the read group: q, k, attn_mask).
    const bool has_bias = args.at(1).refs.size() >= 3;
    std::string shader_name =
        has_bias ? "fused_sdpa_qk_tiled_bias" : "fused_sdpa_qk_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(q));
    add_storage_type_suffix(shader_name, graph->storage_type_of(k));
    add_dtype_suffix(shader_name, graph->dtype_of(q));
    return VK_KERNEL_FROM_STR(shader_name);
  }
}

GlobalWorkGrid pick_sdpa_qk_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPADims d = compute_sdpa_dims(*graph, q, k, input_pos_symint, mode);

  // Dispatch grid: (context_len tiles, S tiles, H * B).
  const uint32_t N4 = utils::div_up_4(static_cast<uint32_t>(d.context_len));
  const uint32_t M4 = utils::div_up_4(static_cast<uint32_t>(d.S));
  const utils::uvec3 extents{N4, M4, static_cast<uint32_t>(d.H * d.B)};
  if (shader.kernel_name.find("_coop") != std::string::npos) {
    return GlobalWorkGrid(extents, kTiledWorkGrid, LocalWorkGroup(1u, 64u, 1u));
  }
  return GlobalWorkGrid(extents, kTiledWorkGrid);
}

LocalWorkGroup pick_sdpa_qk_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  if (gwg.required_lwg_size().is_valid()) {
    return pick_required_lwg(graph, shader, gwg, args, resize_args);
  }
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    return pick_xy_square_lwg(graph, shader, gwg, args, resize_args);
  }
  return default_pick_lwg(graph, shader, gwg, args, resize_args);
}

GlobalWorkGrid pick_sdpa_softmax_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  // LLM reads H from axis -2, fused from axis -3 (handled by
  // compute_sdpa_dims).
  const int64_t num_q_heads = (mode == SDPAMode::LLM)
      ? graph->size_at<int64_t>(-2, q)
      : graph->size_at<int64_t>(-3, q);
  const int64_t seq_len = (mode == SDPAMode::LLM)
      ? graph->size_at<int64_t>(-3, q)
      : graph->size_at<int64_t>(-2, q);
  const int64_t B =
      (mode == SDPAMode::LLM) ? 1 : graph->size_at<int64_t>(-4, q);
  return GlobalWorkGrid(
      {1u,
       static_cast<uint32_t>(seq_len),
       static_cast<uint32_t>(num_q_heads * B)},
      kTiledWorkGrid,
      LocalWorkGroup(64u, 1u, 1u));
}

vkapi::ShaderInfo pick_sdpa_av_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    const ValueRef out = args.at(0).refs.at(0);
    const ValueRef v_cache = args.at(1).refs.at(1);
    const ValueRef q_projected = resize_args.at(0);
    const bool is_gemv = is_single_token(graph, q_projected);

    // Test-only knob (see SDPA.h): -1 auto, 0 forces per-query-head, 1 forces
    // the GQA-reuse shader.
    const ValueRef shader_override = resize_args.at(4);

    std::string shader_name = "sdpa_compute_out";
    if (is_gemv) {
      const int64_t num_q_heads = graph->size_at<int64_t>(-2, q_projected);
      const int64_t num_kv_heads = graph->size_at<int64_t>(-2, v_cache);
      if (resolve_use_gqa(graph, shader_override, num_q_heads, num_kv_heads)) {
        // Grouped-query attention: one workgroup computes all G = Hq / Hkv
        // query heads that share a KV head, loading each V texel once and
        // reusing it across the group. Cuts the dominant V-cache traffic ~Gx
        // for this bandwidth-bound kernel.
        shader_name += "_gqa_coop";
        // The _tile2 (head_dim output-tiled) variant amortizes the attn-weight
        // loads and shared-memory reduction over more outputs per workgroup. A
        // consistent win on Adreno (AV ~1.14-1.63x) but a regression on Mali at
        // common decode contexts (~0.67-0.86x, interleaved median), so it is
        // selected vendor-adaptively (tests can pin it via shader_override; see
        // resolve_use_tile2). pick_sdpa_av_gwg keys the x-dim
        // collapse off this same _tile2 suffix.
        if (resolve_use_tile2(graph, shader_override)) {
          shader_name += "_tile2";
        }
      } else {
        shader_name += "_coop";
      }
    } else {
      shader_name += "_tiled";
    }
    add_storage_type_suffix(shader_name, graph->storage_type_of(out));
    add_storage_type_suffix(shader_name, graph->storage_type_of(v_cache));
    add_dtype_suffix(shader_name, graph->dtype_of(out));
    return VK_KERNEL_FROM_STR(shader_name);
  } else {
    const ValueRef out = args.at(0).refs.at(0);
    const ValueRef v = args.at(1).refs.at(1);
    std::string shader_name = "fused_sdpa_av_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(out));
    add_storage_type_suffix(shader_name, graph->storage_type_of(v));
    add_dtype_suffix(shader_name, graph->dtype_of(out));
    return VK_KERNEL_FROM_STR(shader_name);
  }
}

GlobalWorkGrid pick_sdpa_av_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPADims d = compute_sdpa_dims(*graph, q, k, input_pos_symint, mode);

  const uint32_t N4 = utils::div_up_4(static_cast<uint32_t>(d.D));
  const uint32_t M4 = utils::div_up_4(static_cast<uint32_t>(d.S));

  // The GQA-reuse AV coop shader assigns one workgroup per (d4, kv_h) and emits
  // all G query heads in the group, so the z-dim is the number of KV heads
  // rather than the number of Q heads. d.H is the Q-head count; recover the KV
  // count from the V cache (args read group: [attn_weights_softmax, v]).
  if (shader.kernel_name.find("_gqa_coop") != std::string::npos) {
    const ValueRef v = args.at(1).refs.at(1);
    const uint32_t num_kv_heads = graph->size_at<uint32_t>(-2, v);
    // The _tile2 variant has each workgroup own 2 head_dim texels, so its x-dim
    // collapses to div_up(D4, 2). The plain (TILE_N4 == 1) variant keeps x ==
    // D4.
    const uint32_t x_dim =
        shader.kernel_name.find("_tile2") != std::string::npos
        ? utils::div_up(N4, 2u)
        : N4;
    return GlobalWorkGrid(
        {x_dim, M4, static_cast<uint32_t>(num_kv_heads * d.B)},
        kTiledWorkGrid,
        LocalWorkGroup(1u, 64u, 1u));
  }

  const utils::uvec3 extents{N4, M4, static_cast<uint32_t>(d.H * d.B)};
  if (shader.kernel_name.find("_coop") != std::string::npos) {
    return GlobalWorkGrid(extents, kTiledWorkGrid, LocalWorkGroup(1u, 64u, 1u));
  }
  return GlobalWorkGrid(extents, kTiledWorkGrid);
}

LocalWorkGroup pick_sdpa_av_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  if (gwg.required_lwg_size().is_valid()) {
    return pick_required_lwg(graph, shader, gwg, args, resize_args);
  }
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    return pick_xy_square_lwg(graph, shader, gwg, args, resize_args);
  }
  return default_pick_lwg(graph, shader, gwg, args, resize_args);
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
      kv_cache_update_gwg,
      default_pick_lwg,
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

// Unified QK node (attn_weights = scale * Q @ K^T [+ bias]).
// LLM: pass input_pos_symint, and attn_mask = a 2D [S, C] additive mask that
//      replaces the causal mask, or kDummyValueRef. With a mask, sdpa_impl
//      passes a C - S symint in place of the real input_pos.
// FUSED: pass input_pos_symint = kDummyValueRef, attn_mask = valid ref or
//        kDummyValueRef to indicate no bias.
// scale_val is always passed as a spec const; callers default it to
// 1/sqrt(head_dim) and use the op's scale argument when there is one.
// Shared by add_sdpa_compute_attn_weights_node and the LLM attention with a
// fused decode node, which passes its own global work group picker.
void add_sdpa_compute_attn_weights_node_impl(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask,
    const float scale_val,
    const ValueRef attn_weights,
    const SDPAMode mode,
    const DynamicDispatchNode::PickGwgFn& pick_gwg,
    const ValueRef shader_override) {
  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q),
      graph.sizes_ubo(k),
  };
  std::vector<ValueRef> read_inputs = {q, k};

  if (mode == SDPAMode::LLM) {
    param_ubos.append(graph.get_or_create_int_param_buffer(input_pos_symint));
  }
  if (is_valid(attn_mask)) {
    param_ubos.append(graph.sizes_ubo(attn_mask));
    read_inputs.push_back(attn_mask);
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  // The LLM SDPA's 2D mask with fp16 buffers: the shared-memory tiled kernel.
  // A masked attention never has a fused decode node beside it.
  if (mode == SDPAMode::LLM && is_valid(attn_mask) &&
      graph.dim_of(attn_mask) == 2 && graph.is_buffer_storage(q) &&
      graph.is_buffer_storage(k) && graph.is_buffer_storage(attn_weights) &&
      graph.is_buffer_storage(attn_mask) && graph.dtype_of(q) == vkapi::kHalf &&
      graph.dtype_of(k) == vkapi::kHalf &&
      graph.dtype_of(attn_mask) == vkapi::kHalf &&
      graph.size_at<int32_t>(-1, q) % 16 == 0) {
    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        VK_KERNEL_FROM_STR("sdpa_qk_lds_half"),
        [](ComputeGraph* g,
           const vkapi::ShaderInfo& shader,
           const std::vector<ArgGroup>& a,
           const std::vector<ValueRef>& r) -> GlobalWorkGrid {
          (void)shader;
          (void)r;
          const uint32_t heads = g->size_at<uint32_t>(-2, a.at(1).refs.at(0));
          const uint32_t S = g->size_at<uint32_t>(-3, a.at(1).refs.at(0));
          const uint32_t C = g->size_at<uint32_t>(-3, a.at(1).refs.at(1));
          return GlobalWorkGrid(
              utils::uvec3{
                  utils::div_up(utils::div_up_4(C), 16u) * 8u,
                  utils::div_up(S, 64u) * 8u,
                  heads},
              kExplicitWorkGrid);
        },
        [](ComputeGraph* g,
           const vkapi::ShaderInfo& shader,
           const GlobalWorkGrid& global,
           const std::vector<ArgGroup>& a,
           const std::vector<ValueRef>& r) -> LocalWorkGroup {
          (void)g;
          (void)shader;
          (void)global;
          (void)a;
          (void)r;
          return fixed_lwg(utils::uvec3{8u, 8u, 1u});
        },
        {{attn_weights, vkapi::kWrite}, {read_inputs, vkapi::kRead}},
        param_ubos,
        {},
        {scale_val},
        {q, k, input_pos_symint, mode_ref, shader_override},
        resize_sdpa_attn_weights_node));
    return;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_qk_shader,
      pick_gwg,
      pick_sdpa_qk_lwg,
      // Inputs and Outputs
      {{attn_weights, vkapi::kWrite}, {read_inputs, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {scale_val},
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode,
      //               shader_override_or_dummy]
      {q, k, input_pos_symint, mode_ref, shader_override},
      // Resizing Logic
      resize_sdpa_attn_weights_node));
}

void add_sdpa_compute_attn_weights_node(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask,
    const float scale_val,
    const ValueRef attn_weights,
    const SDPAMode mode) {
  add_sdpa_compute_attn_weights_node_impl(
      graph,
      q,
      k,
      input_pos_symint,
      attn_mask,
      scale_val,
      attn_weights,
      mode,
      pick_sdpa_qk_gwg,
      kDummyValueRef);
}

// Shared by add_sdpa_attn_weights_softmax_node and the LLM attention with a
// fused decode node, which passes its own global work group picker.
void add_sdpa_attn_weights_softmax_node_impl(
    ComputeGraph& graph,
    const ValueRef attn_weights,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax,
    const SDPAMode mode,
    const DynamicDispatchNode::PickGwgFn& pick_gwg,
    const ValueRef shader_override) {
  std::string shader_name;
  if (mode == SDPAMode::LLM) {
    shader_name = "sdpa_attn_weights_softmax";
    add_storage_type_suffix(
        shader_name, graph.storage_type_of(attn_weights_softmax));
    add_dtype_suffix(shader_name, graph.dtype_of(attn_weights_softmax));
  } else {
    shader_name = "fused_sdpa_softmax";
    add_storage_type_suffix(
        shader_name, graph.storage_type_of(attn_weights_softmax));
    add_dtype_suffix(shader_name, graph.dtype_of(attn_weights_softmax));
  }

  vkapi::ParamsBindList param_ubos;
  if (mode == SDPAMode::LLM) {
    param_ubos = {
        graph.sizes_ubo(q),
        graph.sizes_ubo(k),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  } else {
    param_ubos = {graph.sizes_ubo(q), graph.sizes_ubo(k)};
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name),
      pick_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{attn_weights_softmax, vkapi::kWrite}, {attn_weights, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode,
      //               shader_override_or_dummy]
      {q, k, input_pos_symint, mode_ref, shader_override},
      // Resizing Logic
      resize_sdpa_attn_weights_softmax_node));
}

void add_sdpa_attn_weights_softmax_node(
    ComputeGraph& graph,
    const ValueRef attn_weights,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax,
    const SDPAMode mode) {
  add_sdpa_attn_weights_softmax_node_impl(
      graph,
      attn_weights,
      q,
      k,
      input_pos_symint,
      attn_weights_softmax,
      mode,
      pick_sdpa_softmax_gwg,
      kDummyValueRef);
}

//
// Fused decode QK+softmax
//
// An LLM attention without an attn_mask also gets a decode-only node that
// computes Q @ K^T and the softmax in one dispatch. All three nodes are built,
// because the sequence length is only known once sizes are current: a
// dynamic-shape export reports the max bound at graph build time, never 1.
// Their global work group pickers are re-run from
// DynamicDispatchNode::trigger_resize, so the choice below is re-made every
// token, and the nodes not serving the current step get a zero grid, which
// DispatchNode::encode skips. A zero grid still carries the node's local work
// group, since the lwg picker requires one. All three nodes take the resize
// args [q, k, input_pos_symint, mode, shader_override_or_dummy].
//

// Whether the fused node, rather than the separate attn-weights and softmax
// pair, serves the current step.
bool use_fused_qk_softmax(
    ComputeGraph* graph,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const ValueRef shader_override = resize_args.at(4);
  if (!is_single_token(graph, q)) {
    return false;
  }
  if (shader_override != kDummyValueRef &&
      graph->extract_scalar<int64_t>(shader_override) ==
          kShaderOverrideForceSeparateQkSoftmax) {
    return false;
  }
  const int64_t context_len =
      compute_sdpa_dims(*graph, q, k, input_pos_symint, SDPAMode::LLM)
          .context_len;
  return context_len <= kFusedSoftmaxMaxContextLen;
}

vkapi::ShaderInfo pick_sdpa_qk_softmax_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef q = args.at(1).refs.at(0);
  const ValueRef k = args.at(1).refs.at(1);

  std::string shader_name = "sdpa_compute_attn_weights_coop_softmax";
  add_storage_type_suffix(shader_name, graph->storage_type_of(q));
  add_storage_type_suffix(shader_name, graph->storage_type_of(k));
  add_dtype_suffix(shader_name, graph->dtype_of(q));
  return VK_KERNEL_FROM_STR(shader_name);
}

GlobalWorkGrid pick_sdpa_qk_softmax_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  const LocalWorkGroup lwg(64u, 1u, 1u);
  if (!use_fused_qk_softmax(graph, resize_args)) {
    return GlobalWorkGrid({0u, 0u, 0u}, kTiledWorkGrid, lwg);
  }
  const ValueRef q = resize_args.at(0);
  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q);
  return GlobalWorkGrid({1u, seq_len, num_q_heads}, kTiledWorkGrid, lwg);
}

// The separate pair stands down whenever the fused node serves the step.
GlobalWorkGrid pick_sdpa_qk_gwg_beside_fused_decode(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  if (use_fused_qk_softmax(graph, resize_args)) {
    // The coop shader's local work group
    return GlobalWorkGrid(
        {0u, 0u, 0u}, kTiledWorkGrid, LocalWorkGroup(1u, 64u, 1u));
  }
  return pick_sdpa_qk_gwg(graph, shader, args, resize_args);
}

GlobalWorkGrid pick_sdpa_softmax_gwg_beside_fused_decode(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  if (use_fused_qk_softmax(graph, resize_args)) {
    return GlobalWorkGrid(
        {0u, 0u, 0u}, kTiledWorkGrid, LocalWorkGroup(64u, 1u, 1u));
  }
  return pick_sdpa_softmax_gwg(graph, shader, args, resize_args);
}

// Adds the attn-weights and softmax nodes of an LLM attention, and the fused
// decode node unless there is an attn_mask, which the fused shader does not
// apply.
void add_sdpa_llm_attn_weights_softmax_nodes(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask,
    const float scale_val,
    const ValueRef attn_weights,
    const ValueRef attn_weights_softmax,
    const ValueRef shader_override) {
  const bool has_fused_decode_node = !is_valid(attn_mask);
  if (!has_fused_decode_node) {
    add_sdpa_compute_attn_weights_node_impl(
        graph,
        q,
        k,
        input_pos_symint,
        attn_mask,
        scale_val,
        attn_weights,
        SDPAMode::LLM,
        pick_sdpa_qk_gwg,
        shader_override);
    add_sdpa_attn_weights_softmax_node_impl(
        graph,
        attn_weights,
        q,
        k,
        input_pos_symint,
        attn_weights_softmax,
        SDPAMode::LLM,
        pick_sdpa_softmax_gwg,
        shader_override);
    return;
  }

  const ValueRef mode_ref = static_cast<ValueRef>(SDPAMode::LLM);

  // Writes normalized weights straight to attn_weights_softmax, removing the
  // softmax dispatch and the round trip through attn_weights.
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_qk_softmax_shader,
      pick_sdpa_qk_softmax_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{attn_weights_softmax, vkapi::kWrite}, {{q, k}, vkapi::kRead}},
      // Shader param buffers
      {graph.sizes_ubo(q),
       graph.sizes_ubo(k),
       graph.get_or_create_int_param_buffer(input_pos_symint)},
      // Push Constants
      {},
      // Specialization Constants
      {scale_val},
      // Resize Args
      {q, k, input_pos_symint, mode_ref, shader_override},
      // Resizing Logic: the output has the attn_weights shape, and args[1]
      // holds {q, k}, not an attn_weights tensor to copy the shape from.
      resize_sdpa_attn_weights_node));

  add_sdpa_compute_attn_weights_node_impl(
      graph,
      q,
      k,
      input_pos_symint,
      /*attn_mask=*/kDummyValueRef,
      scale_val,
      attn_weights,
      SDPAMode::LLM,
      pick_sdpa_qk_gwg_beside_fused_decode,
      shader_override);

  add_sdpa_attn_weights_softmax_node_impl(
      graph,
      attn_weights,
      q,
      k,
      input_pos_symint,
      attn_weights_softmax,
      SDPAMode::LLM,
      pick_sdpa_softmax_gwg_beside_fused_decode,
      shader_override);
}

void add_sdpa_compute_out_node(
    ComputeGraph& graph,
    const ValueRef attn_weights_softmax,
    const ValueRef v,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef out,
    const SDPAMode mode,
    const ValueRef shader_override) {
  vkapi::ParamsBindList param_ubos;
  if (mode == SDPAMode::LLM) {
    param_ubos = {
        graph.sizes_ubo(q),
        graph.sizes_ubo(v),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  } else {
    param_ubos = {graph.sizes_ubo(q), graph.sizes_ubo(v)};
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  // The GQA-reuse AV coop shader needs its group size G = Hq / Hkv as a
  // specialization constant. The shader declares spec consts [inv_scale (unused
  // in AV), group_size]; pass both positionally so group_size lands at the
  // right slot. Non-GQA AV shaders declare only inv_scale (or none), so the
  // trailing group_size entry is simply ignored there. The GQA gate is fixed at
  // build time (Hq/Hkv/capability don't change across decode steps), matching
  // pick_sdpa_av_shader.
  vkapi::SpecVarList spec_vars = {};
  if (mode == SDPAMode::LLM) {
    const int64_t num_q_heads = graph.size_at<int64_t>(-2, q);
    const int64_t num_kv_heads = graph.size_at<int64_t>(-2, v);
    if (resolve_use_gqa(&graph, shader_override, num_q_heads, num_kv_heads)) {
      const int32_t group_size =
          utils::safe_downcast<int32_t>(num_q_heads / num_kv_heads);
      spec_vars = {1.0f, group_size};
    }
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_av_shader,
      pick_sdpa_av_gwg,
      pick_sdpa_av_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{attn_weights_softmax, v}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      spec_vars,
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode, shader_override]
      {q, k, input_pos_symint, mode_ref, shader_override},
      // Resizing Logic
      resize_sdpa_out_node));
}

// Checks that an additive attn_mask can be bound by the SDPA QK shaders, which
// read it in q's storage type. A buffer mask is read as contiguous rows in q's
// dtype. A texture mask is sampled as floats, so it may be Float or Half
// whatever q's dtype is.
void check_sdpa_attn_mask(
    ComputeGraph& graph,
    const ValueRef attn_mask,
    const ValueRef q) {
  VK_CHECK_COND(
      graph.storage_type_of(attn_mask) == graph.storage_type_of(q),
      "SDPA requires the attn_mask in the query's storage");
  if (graph.is_buffer_storage(attn_mask)) {
    VK_CHECK_COND(
        graph.is_contiguous_buffer_tensor(attn_mask),
        "SDPA requires a contiguous buffer attn_mask");
    VK_CHECK_COND(
        graph.dtype_of(attn_mask) == graph.dtype_of(q),
        "SDPA requires a buffer attn_mask of the query's dtype");
  } else {
    VK_CHECK_COND(
        graph.is_contiguous_texture_tensor(attn_mask),
        "SDPA requires a width packed attn_mask texture");
    VK_CHECK_COND(
        graph.dtype_of(attn_mask) == vkapi::kFloat ||
            graph.dtype_of(attn_mask) == vkapi::kHalf,
        "SDPA requires a Float or Half attn_mask texture");
  }
}

void check_attn_mask_sizes(
    ComputeGraph& graph,
    const ValueRef attn_mask,
    const ValueRef q,
    const ValueRef k) {
  VK_CHECK_COND(
      graph.size_at<int64_t>(-2, attn_mask) == graph.size_at<int64_t>(-3, q) &&
          graph.size_at<int64_t>(-1, attn_mask) ==
              graph.size_at<int64_t>(-3, k),
      "attn_mask must have shape [S, C] = [",
      graph.size_at<int64_t>(-3, q),
      ", ",
      graph.size_at<int64_t>(-3, k),
      "], got [",
      graph.size_at<int64_t>(-2, attn_mask),
      ", ",
      graph.size_at<int64_t>(-1, attn_mask),
      "]");
}

// For a masked LLM SDPA, sets out_symint = C - S (the number of keys in k
// minus the number of queries), which the SDPA shaders read in place of
// input_pos so that they attend to every key. Also checks that the mask still
// has shape [S, C] after a resize, since the shaders would otherwise silently
// skip its missing rows or columns.
void resize_attn_mask_input_pos_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef attn_mask = resize_args.at(2);
  const ValueRef out_symint = resize_args.at(3);
  check_attn_mask_sizes(*graph, attn_mask, q, k);
  graph->set_symint(
      out_symint,
      graph->size_at<int32_t>(-3, k) - graph->size_at<int32_t>(-3, q));
}

ValueRef add_attn_mask_input_pos_node(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef attn_mask) {
  const ValueRef out_symint = graph.add_symint(
      graph.size_at<int32_t>(-3, k) - graph.size_at<int32_t>(-3, q));
  graph.execute_nodes().emplace_back(new ExecuteNode(
      resize_attn_mask_input_pos_node, {q, k, attn_mask, out_symint}));
  return out_symint;
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

  // As in the CPU op, an attn_mask replaces the causal mask: it is a 2D
  // [S, C] additive mask over every key in k_cache, and cannot be combined
  // with is_causal. Without one, the attention is assumed to be causal. The
  // partitioner keeps other configurations on CPU.
  const bool has_mask = graph.val_is_not_none(attn_mask);
  ValueRef mask = kDummyValueRef;
  if (has_mask) {
    VK_CHECK_COND(
        graph.val_is_none(is_causal) || !graph.extract_scalar<bool>(is_causal),
        "attn_mask and is_causal cannot both be set");
    VK_CHECK_COND(graph.dim_of(attn_mask) == 2, "attn_mask must be 2D");
    // A constant mask arrives as a TensorRef, since the fused op handles its
    // own prepacking. Its dtype is checked after prepacking, which applies
    // the same Half -> Float fallback as q on devices without fp16 buffers.
    mask = graph.val_is_tref(attn_mask)
        ? prepack_standard(
              graph,
              attn_mask,
              graph.storage_type_of(q_projected),
              utils::kWidthPacked)
        : attn_mask;
    check_sdpa_attn_mask(graph, mask, q_projected);
    check_attn_mask_sizes(graph, mask, q_projected, k_cache);
  } else {
    VK_CHECK_COND(
        graph.val_is_none(is_causal) || graph.extract_scalar<bool>(is_causal),
        "attention without an attn_mask must be causal");
  }

  const int64_t num_q_heads = graph.size_at<int64_t>(-2, q_projected);
  int64_t max_seq_len = graph.size_at<int64_t>(-3, q_projected);
  const int64_t max_context_len = graph.size_at<int32_t>(-3, k_cache);

  const utils::StorageType attn_weights_storage =
      graph.storage_type_of(q_projected);

  // The attn_weights S and context dims are padded up to a multiple of 4 to
  // match resize_sdpa_attn_weights_node(), which virtual_resizes to
  // align_up_4(seq_len)/align_up_4(context_len). The alignment originates from
  // an Adreno 750 (Samsung S24) correctness workaround (D86226134 / PR #15578).
  // With buffer-backed attn_weights the SDPA shaders index each head at a
  // per-head stride of align_up_4(S) * div_up_4(context) texels, so the
  // allocation must reserve that padded extent — otherwise heads past the first
  // few read/write out of bounds. The texture path is immune: it addresses each
  // head as an image layer rather than via a linear stride.
  const int64_t padded_context_len = utils::align_up_4(max_context_len);

  // If using buffer storage for attn weights, ensure the buffer numel limit is
  // not exceeded by the PADDED allocation. Checking the raw (unpadded) sizes is
  // insufficient: padding can push the actual allocation past the limit even
  // when the raw size fits — e.g. max_seq_len=2047 pads to 2048, so
  // 32*2048*2048 lands exactly on maxStorageBufferRange/4 and the shaders index
  // out of bounds, corrupting attention output.
  if (attn_weights_storage == utils::kBuffer) {
    const int64_t max_buffer_numel = graph.max_buffer_numel();
    if (num_q_heads * utils::align_up_4(max_seq_len) * padded_context_len >=
        max_buffer_numel) {
      // Largest max_seq_len whose padded allocation stays under the limit.
      max_seq_len = max_buffer_numel / (num_q_heads * padded_context_len);
      // Round down to a multiple of 4 (so align_up_4 is a no-op below) and keep
      // the allocation strictly below the limit.
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
      utils::align_up_4(max_seq_len),
      padded_context_len};

  TmpTensor attn_weights(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  // Buffer-backed weights are normalized in place: each row is read in full by
  // the invocations that then overwrite it, so a second buffer is not needed.
  std::unique_ptr<TmpTensor> separate_softmax;
  if (attn_weights_storage != utils::kBuffer) {
    separate_softmax = std::make_unique<TmpTensor>(
        &graph,
        attn_weight_full_sizes,
        graph.dtype_of(q_projected),
        attn_weights_storage,
        utils::kWidthPacked);
  }
  const ValueRef attn_weights_softmax =
      separate_softmax ? ValueRef(*separate_softmax) : ValueRef(attn_weights);

  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = graph.val_is_none(scale)
      ? 1.0f / std::sqrt(static_cast<float>(head_dim_size))
      : graph.extract_scalar<float>(scale);

  // The SDPA shaders attend to the first input_pos + S keys. A masked
  // attention attends to all C keys in k_cache, so give them C - S in place of
  // input_pos. No key tile can be skipped, so with a mask every step costs
  // O(C) rather than O(input_pos + S).
  const ValueRef attn_input_pos = has_mask
      ? add_attn_mask_input_pos_node(graph, q_projected, k_cache, mask)
      : input_pos_symint;

  add_sdpa_llm_attn_weights_softmax_nodes(
      graph,
      q_projected,
      k_cache,
      attn_input_pos,
      mask,
      scale_val,
      attn_weights,
      attn_weights_softmax,
      /*shader_override=*/kDummyValueRef);

  add_sdpa_compute_out_node(
      graph,
      attn_weights_softmax,
      v_cache,
      q_projected,
      /*k=*/kDummyValueRef,
      attn_input_pos,
      out,
      SDPAMode::LLM);
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
  const ValueRef out = args[arg_idx];

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
  const ValueRef dropout_p = args[arg_idx++];
  (void)dropout_p;
  const ValueRef is_causal = args[arg_idx++];
  (void)is_causal;
  const ValueRef scale = args[arg_idx++];

  VK_CHECK_COND(
      graph.val_is_none(attn_mask),
      "attn_mask is not supported by compute_attn_weight_with_kv_cache");

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

  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = graph.val_is_none(scale)
      ? 1.0f / std::sqrt(static_cast<float>(head_dim_size))
      : graph.extract_scalar<float>(scale);

  add_sdpa_compute_attn_weights_node(
      graph,
      q_projected,
      k_cache,
      input_pos_symint,
      /*attn_mask=*/kDummyValueRef,
      scale_val,
      out,
      SDPAMode::LLM);
}

//
// Fused SDPA entry point (et_vk.sdpa.default).
//
// Accepts pre-reshaped [B, H, S, D] tensors (DSHB) plus optional additive
// attn_mask and optional scale scalar. No KV cache; this is the general SDPA
// fused op used by non-LLM models.
//
void fused_sdpa_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q = args[arg_idx++];
  const ValueRef k = args[arg_idx++];
  const ValueRef v = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef scale_ref = args[arg_idx++];
  const ValueRef out = args[arg_idx];

  // Validate inputs
  VK_CHECK_COND(graph.dim_of(q) == 4);
  VK_CHECK_COND(graph.dim_of(k) == 4);
  VK_CHECK_COND(graph.dim_of(v) == 4);
  // Head dim must match between Q and K
  VK_CHECK_COND(graph.size_at<int32_t>(-1, q) == graph.size_at<int32_t>(-1, k));
  // K and V must have same sequence length
  VK_CHECK_COND(graph.size_at<int32_t>(-2, k) == graph.size_at<int32_t>(-2, v));
  // All tensors must be width-packed
  VK_CHECK_COND(graph.packed_dim_of(q) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(k) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(v) == WHCN::kWidthDim);

  // Compute scale
  const int32_t head_dim = graph.size_at<int32_t>(-1, q);
  float scale_val;
  if (graph.val_is_none(scale_ref)) {
    scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim));
  } else {
    scale_val = graph.extract_scalar<float>(scale_ref);
  }

  // Resolve attn_mask: a None value is normalized to kDummyValueRef so the
  // unified helpers can branch with a single `is_valid()` check.
  const ValueRef attn_mask_ref =
      graph.val_is_none(attn_mask) ? kDummyValueRef : attn_mask;

  // Get dimensions for intermediate allocation
  const int64_t B = graph.size_at<int64_t>(-4, q);
  const int64_t H = graph.size_at<int64_t>(-3, q);
  const int64_t S = graph.size_at<int64_t>(-2, q);
  const int64_t L = graph.size_at<int64_t>(-2, k);

  std::vector<int64_t> attn_weight_sizes = {B, H, S, L};

  // attn_weights and attn_weights_softmax follow the output's storage so the
  // entire fused SDPA pipeline uses a uniform storage type. attn_weights stays
  // in fp32 for numerical stability of the Q@K^T accumulation.
  const utils::StorageType attn_storage = graph.storage_type_of(out);

  TmpTensor attn_weights(
      &graph,
      attn_weight_sizes,
      vkapi::ScalarType::Float,
      attn_storage,
      utils::kWidthPacked);

  TmpTensor attn_weights_softmax(
      &graph,
      attn_weight_sizes,
      graph.dtype_of(q),
      attn_storage,
      utils::kWidthPacked);

  // Phase 1: Q @ K^T with fp32 accumulation, apply scale and optional bias
  add_sdpa_compute_attn_weights_node(
      graph,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      attn_mask_ref,
      scale_val,
      attn_weights,
      SDPAMode::FUSED);

  // Phase 2: Softmax in fp32, output in input dtype
  add_sdpa_attn_weights_softmax_node(
      graph,
      attn_weights,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      attn_weights_softmax,
      SDPAMode::FUSED);

  // Phase 3: attn_weights_softmax @ V
  add_sdpa_compute_out_node(
      graph,
      attn_weights_softmax,
      v,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      out,
      SDPAMode::FUSED);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
  VK_REGISTER_OP(update_cache.default, update_cache_impl);
  VK_REGISTER_OP(llama.custom_sdpa.default, sdpa_impl);
  VK_REGISTER_OP(
      testing.compute_attn_weight_with_kv_cache.default,
      compute_attn_weight_with_kv_cache_impl);
  VK_REGISTER_OP(et_vk.sdpa.default, fused_sdpa_impl);
}

} // namespace vkcompute
