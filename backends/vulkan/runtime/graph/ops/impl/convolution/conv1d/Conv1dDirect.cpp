/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv1d/Conv1dDirect.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_conv1d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);
  TensorRefPtr weight_ref = graph->get_tref(extra_args.at(0));

  const int64_t stride_size = graph->get_int_list(extra_args.at(1))->at(0);
  const int64_t padding_size = graph->get_int_list(extra_args.at(2))->at(0);
  const int64_t dilation_size = graph->get_int_list(extra_args.at(3))->at(0);

  const std::vector<int64_t>& weight_sizes = weight_ref->sizes;

  const std::vector<int64_t> in_sizes = graph->sizes_of(self);
  const size_t ndim = in_sizes.size();
  std::vector<int64_t> new_out_sizes(ndim);

  const int64_t kernel_size = weight_sizes.at(2);
  const int64_t in_length = in_sizes.at(2);

  new_out_sizes.at(0) = in_sizes.at(0);
  new_out_sizes.at(1) = weight_sizes.at(0);
  new_out_sizes.at(2) = calc_out_size(
      in_length, kernel_size, stride_size, padding_size, dilation_size, false);

  graph->virtual_resize(out, new_out_sizes);
}

struct OutputParams final {
  float out_min;
  float out_max;
};

// Custom global workgroup size function for conv1d
GlobalWorkGrid conv1d_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  return GlobalWorkGrid(
      {// out length
       graph->size_at<uint32_t>(-1, out),
       // out channels
       static_cast<uint32_t>(graph->size_at<int64_t>(-2, out)),
       // out batches
       utils::div_up_4(graph->size_at<uint32_t>(-3, out))},
      kTiledWorkGrid);
}

// Buffer-backed conv1d with stride 1 and one group: conv1d_lds, a shared-memory
// tiled implicit matmul over the weight's rows x the output columns with
// in_channels * kernel_size terms (conv1d_direct for at most 4 rows). The
// optional tensors, `phases` and `crop` are et_vk.conv1d_fused's fusions.
struct Conv1dFusion {
  ValueRef in_snake = kDummyValueRef;
  ValueRef out_snake = kDummyValueRef;
  ValueRef residual = kDummyValueRef;
  int32_t phases = 1;
  int32_t crop = 0;
};

void add_conv1d_lds_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight,
    const ValueRef bias,
    const int32_t dilation_size,
    const int32_t padding_size,
    const Conv1dFusion& fusion,
    const ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn) {
  const ValueRef arg_weight =
      prepack_standard(graph, weight, utils::kBuffer, utils::kWidthPacked);
  const ValueRef arg_bias = prepack_biases(
      graph,
      bias,
      weight,
      /*transposed = */ false,
      utils::kBuffer,
      utils::kWidthPacked);
  auto present = [&](const ValueRef v) {
    return v != kDummyValueRef && !graph.val_is_none(v);
  };
  // Unused bindings get any buffer; the shader does not read them.
  int32_t flags = 0;
  ValueRef in_snake = arg_bias;
  ValueRef out_snake = arg_bias;
  ValueRef residual = in;
  if (present(fusion.in_snake)) {
    in_snake = prepack_standard(
        graph, fusion.in_snake, utils::kBuffer, utils::kWidthPacked);
    flags |= 1;
  }
  if (present(fusion.out_snake)) {
    out_snake = prepack_standard(
        graph, fusion.out_snake, utils::kBuffer, utils::kWidthPacked);
    flags |= 2;
  }
  if (present(fusion.residual)) {
    residual = fusion.residual;
    flags |= 4;
  }

  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight);
  const int64_t rows = weight_sizes.at(0);
  const int32_t kernel_size = static_cast<int32_t>(weight_sizes.at(2));
  const bool direct = rows <= 4 && fusion.phases == 1 && (flags & 6) == 0 &&
      4 + dilation_size * (kernel_size - 1) <= 32;
  const bool short_tile = !direct && rows <= 32;
  std::string kernel_name(
      direct ? "conv1d_direct"
             : (short_tile ? "conv1d_lds_m32" : "conv1d_lds"));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  // Matmul columns: the convolution's own output length.
  const int32_t span = dilation_size * (kernel_size - 1) - 2 * padding_size;
  auto global_wg = [direct, short_tile, span](
                       ComputeGraph* g,
                       const vkapi::ShaderInfo& shader,
                       const std::vector<ArgGroup>& a,
                       const std::vector<ValueRef>& r) -> utils::uvec3 {
    (void)shader;
    (void)r;
    const ValueRef o = a.at(0).refs.at(0);
    const ValueRef i = a.at(1).refs.at(0);
    if (direct) {
      return {utils::div_up(g->size_at<uint32_t>(-1, o), 4u), 1u, 1u};
    }
    const uint32_t columns =
        static_cast<uint32_t>(g->size_at<int32_t>(-1, i) - span);
    const uint32_t rows =
        static_cast<uint32_t>(g->sizes_of(a.at(1).refs.at(1)).at(0));
    if (short_tile) {
      return {
          utils::div_up(columns, 64u) * 16u, utils::div_up(rows, 32u) * 4u, 1u};
    }
    return {
        utils::div_up(columns, 64u) * 8u, utils::div_up(rows, 64u) * 8u, 1u};
  };
  auto local_wg = [direct, short_tile](
                      ComputeGraph* g,
                      const vkapi::ShaderInfo& shader,
                      const utils::uvec3& global,
                      const std::vector<ArgGroup>& a,
                      const std::vector<ValueRef>& r) -> utils::uvec3 {
    (void)g;
    (void)shader;
    (void)global;
    (void)a;
    (void)r;
    if (direct) {
      return {64u, 1u, 1u};
    }
    return short_tile ? utils::uvec3{16u, 4u, 1u} : utils::uvec3{8u, 8u, 1u};
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg,
      local_wg,
      {{out, vkapi::kWrite},
       {{in, arg_weight, arg_bias, in_snake, out_snake, residual},
        vkapi::kRead}},
      {graph.sizes_ubo(in), graph.sizes_ubo(out)},
      {},
      {kernel_size,
       dilation_size,
       padding_size,
       fusion.phases,
       fusion.crop,
       flags},
      resize_args,
      resize_fn));
}

void resize_conv1d_fused_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);
  const std::vector<int64_t>& weight_sizes =
      graph->get_tref(extra_args.at(0))->sizes;
  const int64_t dilation = graph->extract_scalar<int64_t>(extra_args.at(1));
  const int64_t padding = graph->extract_scalar<int64_t>(extra_args.at(2));
  const int64_t phases = graph->extract_scalar<int64_t>(extra_args.at(3));
  const std::vector<int64_t> in_sizes = graph->sizes_of(self);
  const int64_t length = in_sizes.at(2);
  std::vector<int64_t> out_sizes = in_sizes;
  out_sizes.at(1) = weight_sizes.at(0) / phases;
  out_sizes.at(2) = phases > 1
      ? length * phases
      : length + 2 * padding - dilation * (weight_sizes.at(2) - 1);
  graph->virtual_resize(out, out_sizes);
}

// et_vk.conv1d_fused(x, weight, bias, dilation, padding, in_snake, out_snake,
// residual, phases, crop) -> out: conv1d_lds with its fusions; buffers only.
void conv1d_fused(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef in = args.at(0);
  const ValueRef out = args.at(10);
  const int64_t terms =
      graph.size_at<int64_t>(-2, in) * graph.sizes_of(args.at(1)).at(2);
  VK_CHECK_COND(
      graph.size_at<int64_t>(-3, in) == 1 && graph.is_buffer_storage(in) &&
          graph.is_buffer_storage(out) &&
          graph.packed_dim_of(in) == WHCN::kWidthDim &&
          graph.packed_dim_of(out) == WHCN::kWidthDim && terms % 16 == 0,
      "conv1d_fused needs batch 1, width-packed buffers and in_channels * "
      "kernel_size divisible by 16");
  Conv1dFusion fusion;
  fusion.in_snake = args.at(5);
  fusion.out_snake = args.at(6);
  fusion.residual = args.at(7);
  fusion.phases =
      static_cast<int32_t>(graph.extract_scalar<int64_t>(args.at(8)));
  fusion.crop = static_cast<int32_t>(graph.extract_scalar<int64_t>(args.at(9)));
  add_conv1d_lds_node(
      graph,
      in,
      args.at(1),
      args.at(2),
      static_cast<int32_t>(graph.extract_scalar<int64_t>(args.at(3))),
      static_cast<int32_t>(graph.extract_scalar<int64_t>(args.at(4))),
      fusion,
      out,
      {args.at(1), args.at(3), args.at(4), args.at(8)},
      resize_conv1d_fused_node);
}

void add_conv1d_direct_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef out_min,
    const ValueRef out_max,
    const ValueRef out,
    const bool clamp_out) {
  if (graph.is_buffer_storage(out)) {
    const int64_t terms =
        graph.size_at<int64_t>(-2, in) * graph.sizes_of(weight).at(2);
    VK_CHECK_COND(
        !clamp_out && graph.size_at<int64_t>(-3, in) == 1 &&
            graph.get_int(groups) == 1 &&
            graph.get_int_list(stride)->at(0) == 1 &&
            graph.is_buffer_storage(in) &&
            graph.packed_dim_of(in) == WHCN::kWidthDim &&
            graph.packed_dim_of(out) == WHCN::kWidthDim && terms % 16 == 0,
        "buffer conv1d needs batch 1, stride 1, one group, width-packed "
        "buffers and in_channels * kernel_size divisible by 16");
    // Read before adding values: the list pointers must not outlive this.
    const int32_t dilation_size =
        static_cast<int32_t>(graph.get_int_list(dilation)->at(0));
    const int32_t padding_size =
        static_cast<int32_t>(graph.get_int_list(padding)->at(0));
    return add_conv1d_lds_node(
        graph,
        in,
        weight,
        bias,
        dilation_size,
        padding_size,
        Conv1dFusion(),
        out,
        {weight, stride, padding, dilation},
        resize_conv1d_node);
  }
  ValueRef arg_weight = prepack_standard(
      graph,
      weight,
      graph.storage_type_of(out),
      utils::kChannelsPacked,
      /* passthrough = */ false);
  ValueRef arg_bias = prepack_biases(
      graph,
      bias,
      weight,
      /*transposed = */ false,
      /*storage_type = */ utils::kTexture3D,
      /*memory_layout = */ utils::kWidthPacked);

  float out_min_val = 0.0f;
  float out_max_val = 0.0f;
  if (out_min != kDummyValueRef) {
    out_min_val = graph.extract_scalar<float>(out_min);
  }
  if (out_max != kDummyValueRef) {
    out_max_val = graph.extract_scalar<float>(out_max);
  }

  const int64_t groups_val = graph.get_int(groups);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> weight_sizes = graph.sizes_of(arg_weight);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);

  check_conv_args(graph, in, out);

  const int32_t in_channels = utils::safe_downcast<int32_t>(in_sizes.at(1));
  const int32_t out_channels =
      utils::safe_downcast<int32_t>(weight_sizes.at(0));
  const int32_t kernel_size = utils::safe_downcast<int32_t>(weight_sizes.at(2));
  const int32_t stride_size =
      utils::safe_downcast<int32_t>(graph.get_int_list(stride)->at(0));
  const int32_t padding_size =
      utils::safe_downcast<int32_t>(graph.get_int_list(padding)->at(0));
  const int32_t dilation_size =
      utils::safe_downcast<int32_t>(graph.get_int_list(dilation)->at(0));
  const int32_t in_group_size =
      utils::safe_downcast<int32_t>(in_channels / groups_val);
  const int32_t out_group_size =
      utils::safe_downcast<int32_t>(out_channels / groups_val);

  Kernel1dParams kernel_params = {
      kernel_size,
      stride_size,
      padding_size,
      dilation_size,
      in_group_size,
      out_group_size};

  const OutputParams out_params = {out_min_val, out_max_val};

  std::string kernel_name("conv1d");
  if (clamp_out) {
    kernel_name += "_clamp";
  }
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      conv1d_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // Shader params buffers
      {
          graph.logical_limits_ubo(out),
          graph.sizes_ubo(in),
          graph.create_params_buffer(kernel_params),
          graph.create_params_buffer(out_params),
      },
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(in),
       graph.hashed_layout_of(arg_weight),
       graph.hashed_layout_of(arg_bias)},
      // Resize Args
      {weight, stride, padding, dilation},
      // Resizing Logic
      resize_conv1d_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.conv1d_fused.default, conv1d_fused);
}

} // namespace vkcompute
