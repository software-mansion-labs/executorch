/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_instance_norm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  graph->virtual_resize(
      args.at(0).refs.at(0), graph->sizes_of(args.at(1).refs.at(0)));
}

// et_vk.instance_norm(x, weight, bias, eps): each channel of [N, C, L]
// normalized over L, then scaled and shifted per channel (GroupNorm with one
// group per channel), for any L; buffers only.
void instance_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef in = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef bias = args.at(2);
  const ValueRef eps = args.at(3);
  const ValueRef out = args.at(4);
  VK_CHECK_COND(
      graph.is_buffer_storage(in) && graph.is_buffer_storage(out) &&
          graph.packed_dim_of(in) == WHCN::kWidthDim &&
          graph.packed_dim_of(out) == WHCN::kWidthDim,
      "instance_norm needs width-packed buffers");

  const ValueRef arg_weight =
      prepack_standard(graph, weight, utils::kBuffer, utils::kWidthPacked);
  const ValueRef arg_bias =
      prepack_standard(graph, bias, utils::kBuffer, utils::kWidthPacked);
  VK_CHECK_COND(
      graph.dtype_of(arg_weight) == graph.dtype_of(out) &&
          graph.dtype_of(arg_bias) == graph.dtype_of(out),
      "instance_norm needs weight and bias in the input's dtype");

  const float epsilon = graph.extract_scalar<float>(eps);

  std::string kernel_name = "instance_norm";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      // One workgroup of 256 per row.
      [](ComputeGraph* g,
         const vkapi::ShaderInfo& shader,
         const std::vector<ArgGroup>& a,
         const std::vector<ValueRef>& r) -> utils::uvec3 {
        (void)shader;
        (void)r;
        const ValueRef o = a.at(0).refs.at(0);
        const uint32_t rows =
            static_cast<uint32_t>(g->numel_of(o) / g->size_at<int64_t>(-1, o));
        return {256u, rows, 1u};
      },
      [](ComputeGraph* g,
         const vkapi::ShaderInfo& shader,
         const utils::uvec3& global,
         const std::vector<ArgGroup>& a,
         const std::vector<ValueRef>& r) -> utils::uvec3 {
        (void)g;
        (void)shader;
        (void)global;
        (void)a;
        (void)r;
        return {256u, 1u, 1u};
      },
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      {graph.sizes_ubo(in)},
      {PushConstantDataInfo(&epsilon, sizeof(epsilon))},
      {},
      {},
      resize_instance_norm_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.instance_norm.default, instance_norm);
}

} // namespace vkcompute
