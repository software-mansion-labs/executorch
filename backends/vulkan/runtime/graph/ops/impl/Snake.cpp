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

void resize_snake_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  graph->virtual_resize(
      args.at(0).refs.at(0), graph->sizes_of(args.at(1).refs.at(0)));
}

// et_vk.snake(x, params): DAC's Snake activation over [1, C, L] in one pass;
// buffers only.
void snake(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef in = args.at(0);
  const ValueRef params = args.at(1);
  const ValueRef out = args.at(2);
  VK_CHECK_COND(
      graph.is_buffer_storage(in) && graph.is_buffer_storage(out) &&
          graph.packed_dim_of(in) == WHCN::kWidthDim &&
          graph.packed_dim_of(out) == WHCN::kWidthDim,
      "snake needs width-packed buffers");

  const ValueRef arg_params =
      prepack_standard(graph, params, utils::kBuffer, utils::kWidthPacked);

  std::string kernel_name = "snake";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      [](ComputeGraph* g,
         const vkapi::ShaderInfo& shader,
         const std::vector<ArgGroup>& a,
         const std::vector<ValueRef>& r) -> utils::uvec3 {
        (void)shader;
        (void)r;
        const uint32_t numel =
            static_cast<uint32_t>(g->numel_of(a.at(0).refs.at(0)));
        return {utils::div_up(numel, 4u), 1u, 1u};
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
        return {64u, 1u, 1u};
      },
      {{out, vkapi::kWrite}, {{in, arg_params}, vkapi::kRead}},
      {graph.sizes_ubo(in)},
      {},
      {},
      {},
      resize_snake_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.snake.default, snake);
}

} // namespace vkcompute
