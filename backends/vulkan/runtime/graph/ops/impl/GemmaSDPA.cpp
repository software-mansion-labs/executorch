/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Gemma4-flavored fused/read-only SDPA Vulkan ops.
 *
 *   - et_vk::gemma_sdpa_with_kv_cache  (fused: update + sdpa)
 *   - et_vk::gemma_custom_sdpa         (read-only against an
 *                                       already-updated cache)
 *
 * Both forks of llama's SDPA path (backends/vulkan/.../impl/SDPA.cpp).
 * The shader changes (drop scale, optional additive mask) live in
 * gemma_sdpa_compute_attn_weights_{tiled,coop}.glsl. KV-cache update,
 * softmax, and out-compute are reused from the llama path.
 *
 * Shared with llama:
 *   - sdpa_kv_cache_update.glsl + add_sdpa_kv_cache_update_node
 *   - sdpa_attn_weights_softmax.glsl + add_sdpa_attn_weights_softmax_node
 *   - sdpa_compute_out_{tiled,coop}.glsl + add_sdpa_compute_out_node
 *   - resize_* helpers and pick_* WG helpers
 *
 * Differs from llama:
 *   - schema: no dropout_p, no scale, attn_mask + is_causal allowed,
 *     with attn_mask validated against cache.
 *   - compute-attn-weights shader: gemma_*_tiled / gemma_*_coop, with
 *     a `_mask` / `_nomask` variant suffix selected at dispatch time.
 *   - precondition `attn_mask is None && is_causal == True` is relaxed
 *     to "(mask provided) XOR (is_causal)".
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/runtime/platform/log.h>

#include <unordered_map>

namespace vkcompute {

// Iter 63 fix: map cache_data -> realized in-graph mutable Tensor ValueRef,
// per-ComputeGraph. The first SDPA op that touches a cache materializes it
// via add_tensor_like; subsequent ops (in particular gemma_custom_sdpa for
// KV-shared consumer layers) reuse the same realized tensor so that what
// was written by the source layer is what the consumer reads. Without this,
// each consumer layer allocated a fresh uninitialized in-graph tensor and
// produced gibberish after the very first generated token.
//
// Keying: prefer the underlying TensorRef.data pointer so cache buffers
// that the partitioner duplicated across partitions still resolve to the
// same realized in-graph tensor. Fall back to ValueRef int if the input
// is not a TensorRef (already a Tensor).
namespace {
std::unordered_map<const void*, ValueRef>& cache_map_for(ComputeGraph& graph) {
  static std::
      unordered_map<ComputeGraph*, std::unordered_map<const void*, ValueRef>>
          per_graph;
  return per_graph[&graph];
}
ValueRef realize_cache_once(
    ComputeGraph& graph,
    const ValueRef cache_data,
    utils::StorageType storage) {
  // If the input is already a graph Tensor (e.g. the buffer-mutation output
  // threaded by torch.export from an upstream KV update), use it directly —
  // allocating a fresh tensor would consume uninitialized memory.
  if (graph.val_is_tensor(cache_data)) {
    // Already a graph tensor (post-mutation cache from upstream op).
    return cache_data;
  }
  // It's a TensorRef constant — realize once per data pointer, so the
  // same constant referenced from multiple partitions resolves to one
  // in-graph mutable Tensor.
  const void* key = graph.get_tref(cache_data)->data;
  auto& m = cache_map_for(graph);
  auto it = m.find(key);
  if (it != m.end()) {
    return it->second;
  }
  const ValueRef cache =
      graph.add_tensor_like(cache_data, storage, utils::kWidthPacked);
  m.emplace(key, cache);
  return cache;
}
} // namespace

// Forward decls of the helpers that already live in SDPA.cpp. Defined in
// the same translation-unit-pool because both files end up linked into
// vulkan_backend.

bool is_single_token(ComputeGraph* graph, const ValueRef& q_projected);

void resize_compute_attn_weights_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

void resize_sdpa_attn_weights_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

void resize_sdpa_compute_out_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 attn_weight_scale_and_mask_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_sdpa_compute_attn_weights_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_sdpa_compute_attn_weights_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_sdpa_attn_weights_softmax_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_sdpa_attn_weights_softmax_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

vkapi::ShaderInfo pick_sdpa_compute_out_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_sdpa_compute_out_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_sdpa_compute_out_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

void add_sdpa_kv_cache_update_node(
    ComputeGraph& graph,
    const ValueRef input_pos_symint,
    const ValueRef projected,
    const ValueRef cache);

void add_sdpa_attn_weights_softmax_node(
    ComputeGraph& graph,
    const ValueRef attn_weights,
    const ValueRef q_projected,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax);

void add_sdpa_compute_out_node(
    ComputeGraph& graph,
    const ValueRef attn_weights_softmax,
    const ValueRef v_cache,
    const ValueRef q_projected,
    const ValueRef input_pos_symint,
    const ValueRef out,
    const bool force_tiled = false);

void update_cache_impl(ComputeGraph& graph, const std::vector<ValueRef>& args);

// Forward decl from SDPA.cpp. Stage-1 fix: gemma's no-mask compute-attn-weights
// path delegates to Llama's add_sdpa_compute_attn_weights_node with an
// inv_scale_override of 1.0 (gemma applies its own scale upstream). This
// sidesteps a bug in gemma's compiled compute-attn-weights SPIR-V binary.
// Stage-3-W: force_tiled=true dispatches the tiled shader at S=1 to bypass
// the apparent coop-shader SPIR-V bug at decode (odd q_h produces garbage
// when GQA broadcast is in play with S=1 + start_pos>0).
//
// Audio extension (Phase 2B): mask_per_head=true selects the `_phmask` shader
// variant (mask indexed per Q-head). softcap_value > 0.0f selects the
// `_softcap` variant (applies tanh(x/cap)*cap after mask add). These reroute
// the audio_encoder path off the buggy gemma fork compute-attn-weights SPIR-V
// (which has a multi-row corruption bug at row 1+) and onto the
// multi-row-correct Llama shaders.
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
    const float softcap_value = 0.0f);

// Decode-only fused attn-weights + softmax (writes softmax-normalized values
// directly to the consumer-side buffer). Skips the standalone softmax
// dispatch entirely. See SDPA.cpp for details.
void add_sdpa_compute_attn_weights_with_softmax_node(
    ComputeGraph& graph,
    const ValueRef q_projected,
    const ValueRef k_cache,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax,
    const float inv_scale_override = -1.0f,
    const ValueRef attn_mask = kDummyValueRef);

namespace {

// Pick the gemma compute-attn-weights shader. Mirrors
// pick_sdpa_compute_attn_weights_shader, but additionally encodes the
// mask / nomask suffix so the shader cache key differs.
//
// Audio-extension (Phase 2B): when `mask_per_head=true` (mask sizes[1] ==
// n_q_heads), the shader is the `_phmask` variant that indexes mask per
// head. When `has_softcap=true`, the shader is the `_softcap` variant
// (applies `tanh(x/cap)*cap` after mask-add). Variants combine.
struct GemmaShaderPicker {
  bool has_mask;
  bool mask_per_head;
  bool has_softcap;

  vkapi::ShaderInfo operator()(
      ComputeGraph* graph,
      const std::vector<ArgGroup>& args,
      const std::vector<ValueRef>& resize_args) const {
    (void)resize_args;
    const ValueRef q_projected = args.at(1).refs.at(0);
    const ValueRef k_cache = args.at(1).refs.at(1);

    const bool is_gemv = is_single_token(graph, q_projected);

    std::string shader_name = "gemma_sdpa_compute_attn_weights";
    if (is_gemv) {
      shader_name += "_coop";
    } else {
      shader_name += "_tiled";
    }

    // Suffix order must match codegen output: variant suffix is baked into
    // the yaml shader_variants NAME field; codegen appends storage and dtype
    // cross-product AFTER it. Order:
    // <base>_<variant>_<storage>_<storage>_<dtype>.
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
};

// Audio-extension (Phase 2B): gemma compute-attn-weights with optional
// per-head mask and optional softcap. Vision iter185 still uses the Llama
// shader (no softcap, broadcast mask) via add_sdpa_compute_attn_weights_node;
// this helper is invoked only on the audio path (mask_per_head || softcap > 0).
//
// Push constants encode `inv_softcap` (= 1.0 / softcap) and `softcap` for
// the HAS_SOFTCAP shader variants. Inactive variants ignore them.
void add_gemma_compute_attn_weights_node_with_softcap(
    ComputeGraph& graph,
    const ValueRef q_projected,
    const ValueRef k_cache,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask,
    const ValueRef attn_weights,
    const bool mask_per_head,
    const float softcap_value) {
  const bool has_mask = !graph.val_is_none(attn_mask);
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

  GemmaShaderPicker picker{has_mask, mask_per_head, has_softcap};

  // Push constants: pair of floats (softcap, inv_softcap).
  const float inv_softcap = has_softcap ? (1.0f / softcap_value) : 0.0f;
  std::array<float, 2> pcs = {softcap_value, inv_softcap};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      [picker](
          ComputeGraph* g,
          const std::vector<ArgGroup>& a,
          const std::vector<ValueRef>& r) -> vkapi::ShaderInfo {
        return picker(g, a, r);
      },
      [](ComputeGraph* g,
         const vkapi::ShaderInfo&,
         const std::vector<ArgGroup>& a,
         const std::vector<ValueRef>& r) -> utils::uvec3 {
        (void)r;
        const ValueRef q_projected = a.at(1).refs.at(0);
        const ValueRef k_cache = a.at(1).refs.at(1);
        const uint32_t num_q_heads = g->size_at<uint32_t>(-2, q_projected);
        const uint32_t seq_len = g->size_at<uint32_t>(-3, q_projected);
        const uint32_t max_context_len = g->size_at<uint32_t>(-3, k_cache);
        const uint32_t N4 = utils::div_up_4(max_context_len);
        return {N4, seq_len, num_q_heads};
      },
      pick_sdpa_compute_attn_weights_local_wg_size,
      arg_groups,
      param_ubos,
      // Push Constants: (softcap, inv_softcap) — read by HAS_SOFTCAP variant.
      {PushConstantDataInfo(&pcs[0], sizeof(float) * 2)},
      // Specialization Constants — no inv_scale (Gemma uses scale=1).
      {},
      // Resize Args
      {input_pos_symint},
      // Resizing Logic
      resize_compute_attn_weights_node));
}

void add_gemma_compute_attn_weights_node(
    ComputeGraph& graph,
    const ValueRef q_projected,
    const ValueRef k_cache,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask, // may be kDummyValueRef-ish; see below
    const ValueRef attn_weights) {
  const bool has_mask = !graph.val_is_none(attn_mask);

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

  GemmaShaderPicker picker{
      has_mask,
      /*mask_per_head=*/false,
      /*has_softcap=*/false};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      // Wrap picker in a function-compatible lambda. DynamicDispatchNode's
      // ShaderInfo picker is a std::function-like; this matches the
      // existing pattern used in SDPA.cpp.
      [picker](
          ComputeGraph* g,
          const std::vector<ArgGroup>& a,
          const std::vector<ValueRef>& r) -> vkapi::ShaderInfo {
        return picker(g, a, r);
      },
      // Step 23 Bug-1 workaround: dispatch Y per row (not per 4 rows) so
      // tiled shader's TILE_M=1 covers the full sequence.
      // CMD-REUSE optimization: dispatch over max_context_len (a graph-time
      // constant = k_cache.size(-3)) instead of the current context_len
      // (seq_len + input_pos). This keeps the global WG size constant at
      // decode and lets the deferred-command-buffer cache stay live.
      // Bounds checks at the top of gemma_sdpa_compute_attn_weights_*.glsl
      // early-return for c >= context_len.
      [](ComputeGraph* g,
         const vkapi::ShaderInfo&,
         const std::vector<ArgGroup>& a,
         const std::vector<ValueRef>& r) -> utils::uvec3 {
        (void)r;
        const ValueRef q_projected = a.at(1).refs.at(0);
        const ValueRef k_cache = a.at(1).refs.at(1);
        const uint32_t num_q_heads = g->size_at<uint32_t>(-2, q_projected);
        const uint32_t seq_len = g->size_at<uint32_t>(-3, q_projected);
        const uint32_t max_context_len = g->size_at<uint32_t>(-3, k_cache);
        const uint32_t N4 = utils::div_up_4(max_context_len);
        return {N4, seq_len, num_q_heads};
      },
      pick_sdpa_compute_attn_weights_local_wg_size,
      arg_groups,
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants — no inv_scale (Gemma uses scale=1).
      {},
      // Resize Args
      {input_pos_symint},
      // Resizing Logic
      resize_compute_attn_weights_node));
}

void gemma_sdpa_core(
    ComputeGraph& graph,
    const ValueRef q_projected,
    const ValueRef k_cache,
    const ValueRef v_cache,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask,
    const ValueRef is_causal,
    const ValueRef out,
    const float softcap_value = 0.0f) {
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
  // Packed dim must be the head dim
  VK_CHECK_COND(graph.packed_dim_of(q_projected) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(k_cache) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(v_cache) == WHCN::kWidthDim);
  const bool has_mask = !graph.val_is_none(attn_mask);
  // Audio-extension (Phase 2B): allow per-head mask (sizes[1] == n_q_heads).
  // Vision iter185 path stays at sizes[1] == 1 (broadcast over heads).
  bool mask_per_head = false;
  if (has_mask) {
    VK_CHECK_COND(graph.dim_of(attn_mask) == 4);
    VK_CHECK_COND(graph.packed_dim_of(attn_mask) == WHCN::kWidthDim);
    VK_CHECK_COND(
        graph.size_at<int32_t>(-1, attn_mask) >=
        graph.size_at<int32_t>(-3, k_cache));
    const int32_t mask_h = graph.size_at<int32_t>(-3, attn_mask);
    const int32_t q_h = graph.size_at<int32_t>(-2, q_projected);
    VK_CHECK_COND(mask_h == 1 || mask_h == q_h);
    mask_per_head = (mask_h == q_h);
  } else {
    VK_CHECK_COND(
        graph.val_is_none(is_causal) || graph.extract_scalar<bool>(is_causal));
  }
  const bool has_softcap = softcap_value > 0.0f;
  if (has_softcap) {
    VK_CHECK_COND(has_mask);
  }

  const int64_t num_q_heads = graph.size_at<int64_t>(-2, q_projected);
  int64_t max_seq_len = graph.size_at<int64_t>(-3, q_projected);
  const int64_t max_context_len = graph.size_at<int32_t>(-3, k_cache);

  const utils::StorageType attn_weights_storage =
      graph.storage_type_of(q_projected);

  // Bound max_seq_len by the buffer numel cap when using buffer storage.
  if (attn_weights_storage == utils::kBuffer) {
    const int64_t max_buffer_numel = graph.max_buffer_numel();
    if (num_q_heads * max_seq_len * max_context_len >= max_buffer_numel) {
      max_seq_len = max_buffer_numel / (num_q_heads * max_context_len);
      if (max_seq_len % 4 != 0) {
        max_seq_len = (max_seq_len / 4) * 4;
      } else {
        max_seq_len -= 4;
      }
    }
  }

  std::vector<int64_t> attn_weight_full_sizes = {
      1, num_q_heads, max_seq_len, max_context_len};

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

  // Stage-1 production fix: gemma's compiled compute-attn-weights SPIR-V has
  // a multi-row bug at row 1+. KSWAP3 probe confirmed Llama's
  // add_sdpa_compute_attn_weights_node with inv_scale=1.0 produces cos=1.0.
  // The no-mask path is delegated here. The has_mask path retains the gemma
  // shader (HAS_MASK port to Llama's shader is Stage-2).
  // Stage-2 production fix: route the has_mask path through Llama's shader
  // too (HAS_MASK=1 variant) to avoid the gemma SPIR-V multi-row bug.
  // Stage-3-W workaround: force the tiled shader for ALL gemma SDPA
  // dispatches (including S=1 decode) to bypass the apparent coop-shader
  // SPIR-V bug where odd-numbered q_h values produce garbage at decode
  // (S=1 + start_pos>0) with GQA broadcast. Tiled at S=1 is sub-optimal but
  // correct (TILE_M=4 with bounds checks covers S<=4). The proper coop
  // shader fix is tracked as a follow-up.
  //
  // Softmax fusion: at decode (S=1), use the fused matmul+softmax shader
  // that writes softmax-normalized attn_weights directly to the consumer
  // buffer. Skips the standalone softmax dispatch (one per layer per token).
  //
  // Audio-extension (Phase 2B): when the caller requests per-head mask or
  // softcap, route through the EXTENDED Llama compute-attn-weights shader
  // (which now also supports per-head mask + softcap via the
  // `_mask_phmask[_softcap]` variants). The gemma fork shader is bypassed
  // because its compiled SPIR-V has a multi-row corruption bug at row 1+
  // (TILE_M=4 with rows 1,2,3 of each tile corrupted; ~75% bad → cos≈0.40
  // on audio_encoder iter197). Vision iter185 path (mask_h==1, softcap==0)
  // continues to use Llama's shader via the unmodified `else` branch below.
  const bool fuse_softmax = is_single_token(&graph, q_projected);
  const bool needs_audio_variant = mask_per_head || has_softcap;

  if (needs_audio_variant) {
    // Audio path: delegate to Llama's add_sdpa_compute_attn_weights_node
    // with the new mask_per_head / softcap_value flags. The Llama node-adder
    // selects the `_mask_phmask` / `_mask_softcap` / `_mask_phmask_softcap`
    // shader variant and emits the softcap push constants when needed.
    // inv_scale_override=1.0f because audio (like other gemma paths) absorbs
    // 1/sqrt(d) into QKV-norm / pre-scaled Q.
    add_sdpa_compute_attn_weights_node(
        graph,
        q_projected,
        k_cache,
        input_pos_symint,
        attn_weights,
        /*inv_scale_override=*/1.0f,
        /*attn_mask=*/attn_mask,
        /*force_tiled=*/false,
        /*mask_per_head=*/mask_per_head,
        /*softcap_value=*/softcap_value);

    add_sdpa_attn_weights_softmax_node(
        graph,
        attn_weights,
        q_projected,
        input_pos_symint,
        attn_weights_softmax);
  } else if (fuse_softmax) {
    add_sdpa_compute_attn_weights_with_softmax_node(
        graph,
        q_projected,
        k_cache,
        input_pos_symint,
        attn_weights_softmax,
        /*inv_scale_override=*/1.0f,
        /*attn_mask=*/has_mask ? attn_mask : kDummyValueRef);
  } else {
    add_sdpa_compute_attn_weights_node(
        graph,
        q_projected,
        k_cache,
        input_pos_symint,
        attn_weights,
        /*inv_scale_override=*/1.0f,
        /*attn_mask=*/has_mask ? attn_mask : kDummyValueRef,
        /*force_tiled=*/false);

    add_sdpa_attn_weights_softmax_node(
        graph,
        attn_weights,
        q_projected,
        input_pos_symint,
        attn_weights_softmax);
  }

  // Stage-3-W: also force tiled for compute_out at S=1 (same coop SPIR-V
  // bug pattern as compute_attn_weights_coop).
  add_sdpa_compute_out_node(
      graph,
      attn_weights_softmax,
      v_cache,
      q_projected,
      input_pos_symint,
      out,
      /*force_tiled=*/false);
}

} // namespace

//
// High-level op impls
//

// et_vk::gemma_sdpa_with_kv_cache(
//     Tensor query, Tensor key, Tensor value,
//     Tensor(a!) key_cache, Tensor(b!) value_cache,
//     SymInt start_pos, SymInt seq_len,
//     Tensor? attn_mask=None, bool is_causal=False,
//     float softcap_value=0.0) -> Tensor
void gemma_sdpa_with_kv_cache_impl(
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
  const ValueRef is_causal = args[arg_idx++];
  // Audio-extension (Phase 2B): optional softcap_value.
  float softcap_value = 0.0f;
  if (args.size() > static_cast<size_t>(arg_idx + 1)) {
    softcap_value = graph.extract_scalar<float>(args[arg_idx++]);
  }

  // Output
  const ValueRef out = args[arg_idx++];

  (void)sequence_len;

  // Iter 13 fix: force K/V cache tensors to kTexture3D. The shader variants
  // generated by gemma_sdpa_compute_attn_weights_tiled.yaml declare the
  // k_cache binding as sampler3D for the texture3d combos; if we inherit
  // q_projected's storage and that turns out to be kTexture2D, the runtime
  // would bind a VIEW_TYPE_2D image to a 3D sampler slot
  // (VUID-vkCmdDispatch-viewType-07752). Hard-coding kTexture3D matches the
  // shader yaml default and eliminates the view-type mismatch.
  // Iter 66: revert to the Llama-equivalent simple pattern (single
  // add_tensor_like per call, no dedup map). With iter65's per-layer KV
  // cache, every op call gets a unique cache_data anyway, so the helper
  // collapsed to a no-op. This isolates whether the bug is in our cache
  // wiring vs the gemma compute-attn-weights shader.
  utils::StorageType cache_storage = graph.storage_type_of(q_projected);
  const ValueRef k_cache =
      graph.add_tensor_like(k_cache_data, cache_storage, utils::kWidthPacked);
  const ValueRef v_cache =
      graph.add_tensor_like(v_cache_data, cache_storage, utils::kWidthPacked);

  update_cache_impl(graph, {k_projected, k_cache, input_pos_symint, -1});
  update_cache_impl(graph, {v_projected, v_cache, input_pos_symint, -1});

  gemma_sdpa_core(
      graph,
      q_projected,
      k_cache,
      v_cache,
      input_pos_symint,
      attn_mask,
      is_causal,
      out,
      softcap_value);
}

// et_vk::gemma_custom_sdpa(
//     Tensor query, Tensor key_cache, Tensor value_cache,
//     SymInt start_pos, SymInt seq_len,
//     Tensor? attn_mask=None, bool is_causal=False) -> Tensor
//
// Read-only SDPA against a cache that was already updated by an earlier
// KV-source layer in the same forward pass.
void gemma_custom_sdpa_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_cache_data = args[arg_idx++];
  const ValueRef v_cache_data = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef sequence_len = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef is_causal = args[arg_idx++];
  // Audio-extension (Phase 2B): optional softcap_value.
  float softcap_value = 0.0f;
  if (args.size() > static_cast<size_t>(arg_idx + 1)) {
    softcap_value = graph.extract_scalar<float>(args[arg_idx++]);
  }

  const ValueRef out = args[arg_idx++];

  (void)sequence_len;

  utils::StorageType cache_storage = graph.storage_type_of(q_projected);
  const ValueRef k_cache =
      realize_cache_once(graph, k_cache_data, cache_storage);
  const ValueRef v_cache =
      realize_cache_once(graph, v_cache_data, cache_storage);

  // No update_cache_impl calls — by contract the cache already contains
  // the current step's K/V (for text decoder), OR the cache IS the K/V
  // (for vision encoder B5 path).

  gemma_sdpa_core(
      graph,
      q_projected,
      k_cache,
      v_cache,
      input_pos_symint,
      attn_mask,
      is_causal,
      out,
      softcap_value);
}

REGISTER_OPERATORS {
  // The serializer emits OpCall::name() == OpOverload.__name__, which for
  // torch.ops.et_vk.gemma_sdpa_with_kv_cache.default is
  // "gemma_sdpa_with_kv_cache.default" (the et_vk:: namespace is stripped).
  // Register both bare and namespaced spellings so we are robust to either.
  VK_REGISTER_OP(
      gemma_sdpa_with_kv_cache.default, gemma_sdpa_with_kv_cache_impl);
  VK_REGISTER_OP(gemma_custom_sdpa.default, gemma_custom_sdpa_impl);
  VK_REGISTER_OP(
      et_vk.gemma_sdpa_with_kv_cache.default, gemma_sdpa_with_kv_cache_impl);
  VK_REGISTER_OP(et_vk.gemma_custom_sdpa.default, gemma_custom_sdpa_impl);
}

} // namespace vkcompute
