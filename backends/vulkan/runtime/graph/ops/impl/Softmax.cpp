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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>

namespace vkcompute {

using namespace utils;

// This should match the value of MAX_NTHREADS in softmax_buffer.
constexpr uint32_t kSoftmaxBufferMaxNThreads = 256u;

// The largest worker count the buffer dispatch may ask for. The shared arrays
// in the shader are one ceiling, but they are not the only one: a device
// bounds the invocations in a work group (maxComputeWorkGroupInvocations is
// only guaranteed to be 128) and bounds each axis separately. The buffer
// launch puts every worker on the reduction axis and leaves the other two at
// one, so both device limits apply to the worker count directly, and
// overrunning either aborts the dispatch when the work group is validated.
uint32_t softmax_nworkers_cap(ComputeGraph* graph, const int32_t reduce_dim) {
  const vkapi::Adapter* const adapter = graph->context()->adapter_ptr();
  uint32_t cap = kSoftmaxBufferMaxNThreads;
  cap = std::min(cap, adapter->max_compute_workgroup_invocations());
  cap = std::min(cap, adapter->max_compute_workgroup_size()[reduce_dim]);
  // The shader folds the partials as a tree that halves the worker count each
  // step, so the count has to be a power of two for the last step to land on
  // slot 0. Round the cap down to one.
  uint32_t pow2 = 1u;
  while (pow2 * 2u <= cap) {
    pow2 *= 2u;
  }
  return pow2;
}

// Threads co-operating on one softmax row, scaled with the row length. Buffer
// storage only: the texture path uses a different shader and grouping scheme.
// Backport of pytorch/executorch#22349.
uint32_t softmax_nworkers(
    ComputeGraph* graph,
    const ValueRef in,
    const int32_t reduce_dim) {
  const uint32_t cap = softmax_nworkers_cap(graph, reduce_dim);
  // reduce_dim is a WHCN/xyz index (0 = x = last dim) while size_at counts back
  // from the end, so xyz 0 -> -1, 1 -> -2, 2 -> -3.
  const uint32_t extent = graph->size_at<uint32_t>(-(reduce_dim + 1), in);
  // 4 is what this used to be unconditionally; keep it as the floor so short
  // rows dispatch exactly as they did before.
  uint32_t nworkers = std::min(4u, cap);
  while (nworkers * 2u <= cap && nworkers < extent) {
    nworkers *= 2u;
  }
  return nworkers;
}

utils::uvec3 pick_softmax_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const int dim = resize_args.at(0);

  const int64_t ndim = graph->dim_of(in);
  int32_t reduce_dim = normalize(dim, ndim);
  reduce_dim = nchw_dim_to_whcn_dim(reduce_dim, ndim);

  if (graph->is_buffer_storage(out)) {
    utils::uvec3 global_size = {
        graph->size_at<uint32_t>(-1, out),
        graph->size_at<uint32_t>(-2, out),
        graph->size_at<uint32_t>(-3, out) * graph->size_at<uint32_t>(-4, out)};
    global_size[reduce_dim] = 1;
    return global_size;
  }

  utils::uvec3 global_size = graph->logical_limits_of(out);
  global_size[reduce_dim] = 1;
  return global_size;
}

utils::uvec3 pick_softmax_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)global_workgroup_size;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const int dim = resize_args.at(0);

  const int64_t ndim = graph->dim_of(in);
  int32_t reduce_dim = normalize(dim, ndim);
  reduce_dim = nchw_dim_to_whcn_dim(reduce_dim, ndim);

  const uint32_t nworkers_per_group = 4;

  if (graph->is_buffer_storage(out)) {
    // NWORKERS is baked into softmax_buffer as a specialization constant when
    // the node is built, so the launch has to keep using that value.
    // Recomputing it here from the resized extent would launch fewer threads
    // than the shader's tree reduction indexes over, leaving it to read shared
    // memory slots that no thread wrote. Take it from the resize args instead.
    utils::uvec3 local_wg_size{1, 1, 1};
    local_wg_size[reduce_dim] = static_cast<uint32_t>(resize_args.at(1));
    return local_wg_size;
  }

  const int64_t group_dim_xyz =
      graph->extract_scalar<int64_t>(resize_args.at(1));
  const uint32_t ngroups = 4;

  utils::uvec3 local_wg_size{1, 1, 1};
  local_wg_size[reduce_dim] = nworkers_per_group;
  local_wg_size[group_dim_xyz] = ngroups;
  return local_wg_size;
}

void resize_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  graph->virtual_resize(out, in_sizes);
}

void add_softmax_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef out,
    bool log_softmax) {
  const int64_t ndim = graph.dim_of(in);

  int32_t reduce_dim_nchw = graph.extract_scalar<int32_t>(dim_ref);
  reduce_dim_nchw = normalize(reduce_dim_nchw, ndim);
  const int32_t reduce_dim_xyz = nchw_dim_to_whcn_dim(reduce_dim_nchw, ndim);

  // Check that the concat dim is not the reduction dim, if the tensor has a
  // batch dim greater than 1.
  if (graph.dim_of(in) == 4 && graph.size_at<int>(0, in) > 1) {
    VK_CHECK_COND(
        graph.concat_dim_of(in) != reduce_dim_xyz,
        "Softmax shader currently does not support concat dim == reduce dim");
    VK_CHECK_COND(
        graph.concat_dim_of(out) != reduce_dim_xyz,
        "Softmax shader currently does not support concat dim == reduce dim");
  }

  std::string kernel_name = "softmax";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  if (log_softmax) {
    kernel_name = "log_" + kernel_name;
  }

  // This should match the value of MAX_NTHREADS in the softmax shader.
  constexpr uint32_t max_nthreads = 16;
  const uint32_t nworkers_per_group = 4;
  const uint32_t ngroups = 4;
  VK_CHECK_COND(nworkers_per_group * ngroups <= max_nthreads);

  const int dim_val = graph.extract_scalar<int>(dim_ref);

  const int32_t buffer_nworkers = utils::safe_downcast<int32_t>(
      softmax_nworkers(&graph, in, reduce_dim_xyz));
  vkapi::SpecVarList spec_constants = {reduce_dim_xyz, buffer_nworkers};
  std::vector<ValueRef> resize_args = {dim_val, buffer_nworkers};

  if (!graph.is_buffer_storage(out)) {
    const int other_dim_1 = (reduce_dim_xyz + 1) % 3;
    const int other_dim_2 = (reduce_dim_xyz + 2) % 3;
    int32_t group_dim;
    utils::uvec3 global_wg_size = graph.logical_limits_of(out);
    if (global_wg_size[other_dim_1] > global_wg_size[other_dim_2]) {
      group_dim = other_dim_1;
    } else {
      group_dim = other_dim_2;
    }

    spec_constants = {graph.hashed_layout_of(out), reduce_dim_xyz, group_dim};

    const ValueRef group_dim_xyz_ref =
        graph.get_or_add_value_for_int(group_dim);
    resize_args = {dim_val, group_dim_xyz_ref};
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_softmax_global_wg_size,
      pick_softmax_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {graph.meta_ubo(in), graph.meta_ubo(out)},
      // Push Constants
      {},
      // Specialization Constants
      spec_constants,
      // Resize Args
      resize_args,
      // Resizing Logic
      resize_softmax_node));
}

void softmax(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[1] bool half_to_float is unused
  return add_softmax_node(
      graph, args[0], args[1], args[3], /* log_softmax = */ false);
}

void log_softmax(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[1] bool half_to_float is unused
  return add_softmax_node(
      graph, args[0], args[1], args[3], /* log_softmax = */ true);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten._softmax.default, softmax);
  VK_REGISTER_OP(aten._log_softmax.default, log_softmax);
}

} // namespace vkcompute
