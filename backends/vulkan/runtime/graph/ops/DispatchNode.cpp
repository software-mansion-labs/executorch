/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/BindingUtils.h>

#include <cstdio>
#include <cstdlib>

namespace vkcompute {

// Iter 21 instrumentation: env-gated per-dispatch trace.
//   ETVK_DISPATCH_TRACE=<path>  → append one tab-separated line per
//   dispatch:  <kernel_name>\t<seq>\tW=<r,r,...>\tR=<r,r,...>
// where <seq> is a monotonically-incrementing per-process counter and
// W/R lists are the ValueRefs (ints) bound as kWrite / kRead.
namespace {
FILE* dispatch_trace_fp() {
  static FILE* fp = []() -> FILE* {
    const char* path = std::getenv("ETVK_DISPATCH_TRACE");
    if (path == nullptr || *path == '\0') {
      return nullptr;
    }
    FILE* f = std::fopen(path, "w");
    if (f != nullptr) {
      std::setvbuf(f, nullptr, _IOLBF, 0);
    }
    return f;
  }();
  return fp;
}
} // namespace

DispatchNode::DispatchNode(
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const utils::uvec3& local_workgroup_size,
    const std::vector<ArgGroup>& args,
    const vkapi::ParamsBindList& params,
    const std::vector<PushConstantDataInfo>& push_constants,
    const vkapi::SpecVarList& spec_vars,
    const std::vector<ValueRef>& resize_args,
    const ResizeFunction& resize_fn)
    : ExecuteNode(resize_fn, resize_args, args, shader.kernel_name),
      shader_(shader),
      global_workgroup_size_(global_workgroup_size),
      local_workgroup_size_(local_workgroup_size),
      params_(params),
      spec_vars_(spec_vars),
      push_constants_(push_constants) {
  graph.update_descriptor_counts(shader, /*execute = */ true);
}

void DispatchNode::prepare_pipelines(ComputeGraph* graph) {
  graph->register_pipeline_to_create(
      shader_, local_workgroup_size_, spec_vars_, push_constants_);
}

void DispatchNode::encode(ComputeGraph* graph) {
  if (!shader_) {
    return;
  }

  // If any global wg size element is 0, then skip encoding this shader
  if (global_workgroup_size_[0] == 0 || global_workgroup_size_[1] == 0 ||
      global_workgroup_size_[2] == 0) {
    return;
  }

  api::Context* const context = graph->context();
  vkapi::PipelineBarrier pipeline_barrier{};

  context->check_device_capabilities(shader_);

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  write_push_constant_data();

#ifdef ET_EVENT_TRACER_ENABLED
  std::string event_name;
  if (!operator_json.empty()) {
    event_name += "\"operator\": {" + operator_json + "}, ";
  }
  event_name += "\"kernel_name\": \"" + shader_.kernel_name + "\", ";
  event_name += "\"operator_id\": " + std::to_string(operator_count);
#endif

  context->report_shader_dispatch_start(
#ifdef ET_EVENT_TRACER_ENABLED
      event_name,
#else
      shader_.kernel_name,
#endif
      global_workgroup_size_,
      local_workgroup_size_,
      node_id_);

  // Iter 21 trace: terse, one tab-separated line per dispatch.
  if (FILE* fp = dispatch_trace_fp()) {
    static thread_local uint64_t seq = 0;
    std::fprintf(
        fp,
        "%lu\t%s",
        static_cast<unsigned long>(seq++),
        shader_.kernel_name.c_str());
    for (const auto& g : args_) {
      const char* tag = (g.access == vkapi::kReadWrite)
          ? "RW"
          : ((g.access & vkapi::kWrite)
                 ? "W"
                 : ((g.access & vkapi::kRead) ? "R" : "N"));
      std::fprintf(fp, "\t%s=", tag);
      bool first = true;
      for (const auto ref : g.refs) {
        std::fprintf(fp, "%s%d", first ? "" : ",", static_cast<int>(ref));
        first = false;
      }
    }
    std::fprintf(
        fp,
        "\twg=%u,%u,%u\n",
        static_cast<unsigned>(global_workgroup_size_[0]),
        static_cast<unsigned>(global_workgroup_size_[1]),
        static_cast<unsigned>(global_workgroup_size_[2]));
  }

  vkapi::DescriptorSet descriptor_set = context->get_descriptor_set(
      shader_, local_workgroup_size_, spec_vars_, push_constants_offset_);

  uint32_t idx = 0;
  idx = bind_values_to_descriptor_set(
      graph, args_, pipeline_barrier, descriptor_set, idx);

  bind_params_to_descriptor_set(params_, descriptor_set, idx);

  context->register_shader_dispatch(
      descriptor_set,
      pipeline_barrier,
      shader_,
      global_workgroup_size_,
      push_constants_data_.data(),
      push_constants_offset_);

  context->report_shader_dispatch_end();
}

void DispatchNode::write_push_constant_data() {
  push_constants_offset_ = 0;
  for (const auto& push_constant : push_constants_) {
    push_constants_offset_ += push_constant.write(
        push_constants_data_.data(),
        push_constants_offset_,
        kMaxPushConstantSize);
  }
}

bool DispatchNode::trigger_resize(ComputeGraph* graph) {
  const bool any_arg_updated = ExecuteNode::trigger_resize(graph);

  if (any_arg_updated) {
    // If this shader uses push constants, and the tensor metadata associated
    // with the push constants has changed, then the command buffer needs to be
    // re-encoded since push constants cannot be updated.
    for (const auto& push_constant : push_constants_) {
      if (push_constant.is_tensor_metadata() &&
          graph->was_value_updated(push_constant.value())) {
        graph->set_requires_reencode();
      }
    }
  }
  return any_arg_updated;
}

} // namespace vkcompute
