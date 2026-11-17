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

namespace {

// A fixed local work group, never refitted to the global grid.
LocalWorkGroup fixed_lwg(const utils::uvec3& size) {
  return LocalWorkGroup(size, size[0u] * size[1u] * size[2u]);
}

} // namespace

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

// Buffer-backed conv1d of batch 1: conv1d_lds, a shared-memory tiled implicit
// matmul per group over its rows x the output columns with
// in_channels / groups * kernel_size terms; conv1d_direct where that is not a
// multiple of 16 or there are at most 4 rows. The optional tensors, `phases`
// and `crop` are et_vk.conv1d_fused's fusions; the clamp is
// et_vk.conv_with_clamp's.
struct Conv1dFusion {
  ValueRef in_snake = kDummyValueRef;
  ValueRef out_snake = kDummyValueRef;
  ValueRef residual = kDummyValueRef;
  int32_t phases = 1;
  int32_t crop = 0;
  bool clamp = false;
  float out_min = 0.0f;
  float out_max = 0.0f;
};

// et_vk.conv1d_q8's int8 weights: their fp32 group scales, and the geometry
// that the packed [C, terms / 4] weight no longer carries.
struct Conv1dQuant {
  ValueRef scales = kDummyValueRef;
  int32_t kernel_size = 0;
  int32_t group_size = 0;
};

void add_conv1d_lds_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight,
    const ValueRef bias,
    const int32_t stride_size,
    const int32_t dilation_size,
    const int32_t padding_size,
    const int32_t groups,
    const Conv1dFusion& fusion,
    const ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn,
    const Conv1dQuant& quant = Conv1dQuant()) {
  const bool q8 = quant.scales != kDummyValueRef;
  const ValueRef arg_weight =
      prepack_standard(graph, weight, utils::kBuffer, utils::kWidthPacked);
  // int8 weights come with a bias in the activations' dtype; prepack_biases
  // would give it the weight's.
  const ValueRef arg_bias = q8
      ? prepack_standard(graph, bias, utils::kBuffer, utils::kWidthPacked)
      : prepack_biases(
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
  if (fusion.clamp) {
    flags |= 8;
  }

  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight);
  const int64_t rows = weight_sizes.at(0) / groups;
  const int32_t kernel_size =
      q8 ? quant.kernel_size : static_cast<int32_t>(weight_sizes.at(2));
  const int64_t terms =
      q8 ? weight_sizes.at(1) * 4 : weight_sizes.at(1) * kernel_size;
  const bool direct = !q8 && (rows <= 4 || terms % 16 != 0) &&
      fusion.phases == 1 && (flags & 6) == 0 &&
      3 * stride_size + dilation_size * (kernel_size - 1) + 1 <= 64;
  VK_CHECK_COND(
      direct || terms % 16 == 0,
      "conv1d_lds needs in_channels / groups * kernel_size divisible by 16");
  const bool short_tile = !direct && rows <= 32;
  std::string kernel_name(
      direct ? "conv1d_direct"
             : (short_tile ? "conv1d_lds_m32" : "conv1d_lds"));
  if (q8) {
    kernel_name += "_q8";
  }
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  std::vector<ValueRef> reads{in, arg_weight};
  if (q8) {
    reads.push_back(prepack_standard(
        graph, quant.scales, utils::kBuffer, utils::kWidthPacked));
  }
  reads.insert(reads.end(), {arg_bias, in_snake, out_snake, residual});

  // Matmul columns: the convolution's own output length.
  const int32_t span = dilation_size * (kernel_size - 1) - 2 * padding_size;
  auto global_wg = [direct, short_tile, span, stride_size, groups](
                       ComputeGraph* g,
                       const vkapi::ShaderInfo& shader,
                       const std::vector<ArgGroup>& a,
                       const std::vector<ValueRef>& r) -> GlobalWorkGrid {
    (void)shader;
    (void)r;
    const ValueRef o = a.at(0).refs.at(0);
    const ValueRef i = a.at(1).refs.at(0);
    if (direct) {
      return GlobalWorkGrid(
          utils::uvec3{
              utils::div_up(g->size_at<uint32_t>(-1, o), 4u),
              g->size_at<uint32_t>(-2, o),
              1u},
          kExplicitWorkGrid);
    }
    const uint32_t columns = static_cast<uint32_t>(
        (g->size_at<int32_t>(-1, i) - span - 1) / stride_size + 1);
    const uint32_t rows =
        static_cast<uint32_t>(g->sizes_of(a.at(1).refs.at(1)).at(0) / groups);
    const uint32_t layers = static_cast<uint32_t>(groups);
    if (short_tile) {
      return GlobalWorkGrid(
          utils::uvec3{
              utils::div_up(columns, 64u) * 16u,
              utils::div_up(rows, 32u) * 4u,
              layers},
          kExplicitWorkGrid);
    }
    return GlobalWorkGrid(
        utils::uvec3{
            utils::div_up(columns, 64u) * 8u,
            utils::div_up(rows, 64u) * 8u,
            layers},
        kExplicitWorkGrid);
  };
  auto local_wg = [direct, short_tile](
                      ComputeGraph* g,
                      const vkapi::ShaderInfo& shader,
                      const GlobalWorkGrid& global,
                      const std::vector<ArgGroup>& a,
                      const std::vector<ValueRef>& r) -> LocalWorkGroup {
    (void)g;
    (void)shader;
    (void)global;
    (void)a;
    (void)r;
    if (direct) {
      return fixed_lwg(utils::uvec3{64u, 1u, 1u});
    }
    return fixed_lwg(utils::uvec3(
        short_tile ? utils::uvec3{16u, 4u, 1u} : utils::uvec3{8u, 8u, 1u}));
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg,
      local_wg,
      {{out, vkapi::kWrite}, {reads, vkapi::kRead}},
      {graph.sizes_ubo(in), graph.sizes_ubo(out)},
      {},
      {kernel_size,
       dilation_size,
       padding_size,
       fusion.phases,
       fusion.crop,
       flags,
       stride_size,
       groups,
       q8 ? quant.group_size : 4,
       fusion.out_min,
       fusion.out_max},
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
      /*stride_size = */ 1,
      static_cast<int32_t>(graph.extract_scalar<int64_t>(args.at(3))),
      static_cast<int32_t>(graph.extract_scalar<int64_t>(args.at(4))),
      /*groups = */ 1,
      fusion,
      out,
      {args.at(1), args.at(3), args.at(4), args.at(8)},
      resize_conv1d_fused_node);
}

void resize_conv1d_q8_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  std::vector<int64_t> out_sizes = graph->sizes_of(args.at(1).refs.at(0));
  out_sizes.at(1) = graph->get_tref(extra_args.at(0))->sizes.at(0);
  out_sizes.at(2) = calc_out_size(
      out_sizes.at(2),
      graph->extract_scalar<int64_t>(extra_args.at(1)),
      graph->extract_scalar<int64_t>(extra_args.at(2)),
      graph->extract_scalar<int64_t>(extra_args.at(3)),
      graph->extract_scalar<int64_t>(extra_args.at(4)),
      false);
  graph->virtual_resize(out, out_sizes);
}

// et_vk.conv1d_q8(x, weight, scales, bias, kernel_size, stride, padding,
// dilation, groups, in_snake, out_snake, residual) -> out: conv1d_lds on int8
// weights ([C, terms / 4] int32, 4 per word) with fp32 scales
// [C, terms / group_size], plus conv1d_fused's fusions; buffers only.
void conv1d_q8(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef in = args.at(0);
  const ValueRef out = args.at(12);
  VK_CHECK_COND(
      graph.size_at<int64_t>(-3, in) == 1 && graph.is_buffer_storage(in) &&
          graph.is_buffer_storage(out) &&
          graph.packed_dim_of(in) == WHCN::kWidthDim &&
          graph.packed_dim_of(out) == WHCN::kWidthDim,
      "conv1d_q8 needs batch 1 and width-packed buffers");
  auto scalar = [&](const int i) {
    return static_cast<int32_t>(graph.extract_scalar<int64_t>(args.at(i)));
  };
  Conv1dQuant quant;
  quant.scales = args.at(2);
  quant.kernel_size = scalar(4);
  quant.group_size = static_cast<int32_t>(
      graph.sizes_of(args.at(1)).at(1) * 4 / graph.sizes_of(args.at(2)).at(1));
  Conv1dFusion fusion;
  fusion.in_snake = args.at(9);
  fusion.out_snake = args.at(10);
  fusion.residual = args.at(11);
  add_conv1d_lds_node(
      graph,
      in,
      args.at(1),
      args.at(3),
      /*stride_size = */ scalar(5),
      /*dilation_size = */ scalar(7),
      /*padding_size = */ scalar(6),
      /*groups = */ scalar(8),
      fusion,
      out,
      {args.at(1), args.at(4), args.at(5), args.at(6), args.at(7)},
      resize_conv1d_q8_node,
      quant);
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
    VK_CHECK_COND(
        graph.size_at<int64_t>(-3, in) == 1 && graph.is_buffer_storage(in) &&
            graph.packed_dim_of(in) == WHCN::kWidthDim &&
            graph.packed_dim_of(out) == WHCN::kWidthDim,
        "buffer conv1d needs batch 1 and width-packed buffers");
    // Read before adding values: the list pointers must not outlive this.
    const int32_t stride_size =
        static_cast<int32_t>(graph.get_int_list(stride)->at(0));
    const int32_t dilation_size =
        static_cast<int32_t>(graph.get_int_list(dilation)->at(0));
    const int32_t padding_size =
        static_cast<int32_t>(graph.get_int_list(padding)->at(0));
    const int32_t groups_val = static_cast<int32_t>(graph.get_int(groups));
    Conv1dFusion fusion;
    if (clamp_out) {
      fusion.clamp = true;
      fusion.out_min = graph.extract_scalar<float>(out_min);
      fusion.out_max = graph.extract_scalar<float>(out_max);
    }
    return add_conv1d_lds_node(
        graph,
        in,
        weight,
        bias,
        stride_size,
        dilation_size,
        padding_size,
        groups_val,
        fusion,
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
  VK_REGISTER_OP(et_vk.conv1d_q8.default, conv1d_q8);
}

} // namespace vkcompute
