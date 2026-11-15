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

void resize_linear_qgsw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  std::vector<int64_t> out_sizes = graph->sizes_of(args.at(1).refs.at(0));
  out_sizes.back() = graph->extract_scalar<int64_t>(extra_args.at(0));
  graph->virtual_resize(args.at(0).refs.at(0), out_sizes);
}

// et_vk.linear_qgsw(x, weight, scales, bias, group_size, bits) -> [..., N]:
// x [..., K] times int8 or int4 weights stored transposed and packed along N
// (int8: [K, N / 4] int32 words of 4 columns; int4: [K, N / 8] words of 8
// columns, + 8), with fp32 scales [K / group_size, N]; linear_qgsw_lds, on
// buffers.
void linear_qgsw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef in = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef scales = args.at(2);
  const ValueRef bias = args.at(3);
  const int64_t group_size = graph.extract_scalar<int64_t>(args.at(4));
  const int64_t bits = graph.extract_scalar<int64_t>(args.at(5));
  const ValueRef out = args.at(6);
  const int64_t K = graph.size_at<int64_t>(-1, in);
  const int64_t N = graph.size_at<int64_t>(-1, out);
  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight);
  VK_CHECK_COND(
      graph.is_buffer_storage(in) && graph.is_buffer_storage(out) &&
          graph.packed_dim_of(in) == WHCN::kWidthDim &&
          graph.packed_dim_of(out) == WHCN::kWidthDim &&
          (bits == 8 || bits == 4) && K % 16 == 0 && N % 64 == 0 &&
          weight_sizes.at(0) == K && weight_sizes.at(1) * (32 / bits) == N,
      "linear_qgsw needs width-packed buffers, 8 or 4 bits, K % 16 == 0 and "
      "N % 64 == 0");

  const ValueRef arg_weight =
      prepack_standard(graph, weight, utils::kBuffer, utils::kWidthPacked);
  const ValueRef arg_scales =
      prepack_standard(graph, scales, utils::kBuffer, utils::kWidthPacked);
  const bool has_bias = graph.val_is_not_none(bias);
  // Without a bias, the scales stand in for the unread binding.
  const ValueRef arg_bias = has_bias
      ? prepack_standard(graph, bias, utils::kBuffer, utils::kWidthPacked)
      : arg_scales;

  std::string kernel_name =
      bits == 8 ? "linear_qgsw_lds_8bit" : "linear_qgsw_lds_4bit";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      // 8 x 8 threads per 64 x 64 output tile.
      [](ComputeGraph* g,
         const vkapi::ShaderInfo& shader,
         const std::vector<ArgGroup>& a,
         const std::vector<ValueRef>& r) -> utils::uvec3 {
        (void)shader;
        (void)r;
        const ValueRef o = a.at(0).refs.at(0);
        const int64_t n = g->size_at<int64_t>(-1, o);
        const int64_t m = g->numel_of(o) / n;
        return {
            utils::div_up(static_cast<uint32_t>(n), 64u) * 8u,
            utils::div_up(static_cast<uint32_t>(m), 64u) * 8u,
            1u};
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
        return {8u, 8u, 1u};
      },
      {{out, vkapi::kWrite},
       {{in, arg_weight, arg_scales, arg_bias}, vkapi::kRead}},
      {graph.sizes_ubo(in), graph.sizes_ubo(out)},
      {},
      {static_cast<int32_t>(group_size), has_bias ? 1 : 0},
      {graph.add_scalar<int64_t>(N)},
      resize_linear_qgsw_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_qgsw.default, linear_qgsw);
}

} // namespace vkcompute
