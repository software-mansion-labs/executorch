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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_rms_norm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& /*extra_args*/) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  graph->virtual_resize(out, graph->sizes_of(in));
}

utils::uvec3 rms_norm_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& /*shader*/,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& /*resize_args*/) {
  // One workgroup per row. Number of rows = numel / hidden_size.
  const ValueRef in = args.at(1).refs.at(0);
  const std::vector<int64_t> sizes = graph->sizes_of(in);
  const int64_t hidden = sizes.back();
  const int64_t numel = graph->numel_of(in);
  const int64_t num_rows = (hidden > 0) ? (numel / hidden) : 0;
  return {1u, utils::safe_downcast<uint32_t>(num_rows), 1u};
}

utils::uvec3 rms_norm_local_wg_size(
    ComputeGraph* /*graph*/,
    const vkapi::ShaderInfo& /*shader*/,
    const utils::uvec3& /*global_workgroup_size*/,
    const std::vector<ArgGroup>& /*args*/,
    const std::vector<ValueRef>& /*resize_args*/) {
  // Must match NUM_WORKERS_PER_ROW in the shader.
  return {64u, 1u, 1u};
}

void add_rms_norm_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef normalized_shape,
    const ValueRef weight_data,
    const ValueRef eps,
    const ValueRef out) {
  const auto normalized_shape_dim =
      graph.get_int_list(normalized_shape)->size();
  if (normalized_shape_dim > 1) {
    VK_THROW("rms_norm only supports normalized_shape with dim == 1");
  }

  const bool no_weight = graph.val_is_none(weight_data);

  VK_CHECK_COND(
      graph.is_buffer_storage(in) && graph.is_buffer_storage(out),
      "Vulkan rms_norm only supports buffer storage");

  float epsilon = graph.extract_scalar<float>(eps);

  VK_CHECK_COND(check_same_packed_dim(graph, in, out));

  std::string kernel_name(no_weight ? "rms_norm_no_weight_buffer"
                                    : "rms_norm_buffer");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(out),
      graph.meta_ubo(in),
  };

  std::vector<ValueRef> read_refs;
  read_refs.push_back(in);
  if (!no_weight) {
    ValueRef arg_weight = prepack_standard_like(graph, weight_data, in);
    read_refs.push_back(arg_weight);
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      rms_norm_global_wg_size,
      rms_norm_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {read_refs, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {PushConstantDataInfo(&epsilon, sizeof(epsilon))},
      // Specialization Constants
      {},
      // Resize Args
      {normalized_shape},
      // Resizing Logic
      resize_rms_norm_node));
}

void rms_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Signature: rms_norm(input, normalized_shape, weight, eps) -> out
  return add_rms_norm_node(graph, args[0], args[1], args[2], args[3], args[4]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.rms_norm.default, rms_norm);
}

} // namespace vkcompute
