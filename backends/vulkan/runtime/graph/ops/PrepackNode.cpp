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
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

namespace vkcompute {

PrepackNode::PrepackNode(
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const utils::uvec3& local_workgroup_size,
    const ValueRef tref,
    const ValueRef packed,
    const vkapi::ParamsBindList& params,
    const vkapi::SpecVarList& spec_vars,
    const std::vector<PushConstantDataInfo>& push_constants)
    : shader_(shader),
      global_workgroup_size_(global_workgroup_size),
      local_workgroup_size_(local_workgroup_size),
      tref_(tref),
      packed_(packed),
      params_(params),
      spec_vars_(spec_vars),
      push_constants_(push_constants) {
  graph.update_descriptor_counts(shader, /*execute = */ false);
  if (!graph.val_is_none(tref)) {
    graph.get_tref(tref)->prepack_use_count++;
  }
}

api::StagingBuffer PrepackNode::create_staging_buffer(ComputeGraph* graph) {
  // If no TensorRef is provided, create a staging buffer of zeros based on the
  // Tensor metadata.
  if (graph->val_is_none(tref_)) {
    const std::vector<int64_t> packed_sizes = graph->sizes_of(packed_);
    size_t numel = utils::multiply_integers(packed_sizes);
    api::StagingBuffer staging(
        graph->context(),
        graph->dtype_of(packed_),
        numel,
        vkapi::CopyDirection::HOST_TO_DEVICE);
    staging.set_staging_zeros();
    return staging;
  }

  TensorRefPtr tref = graph->get_tref(tref_);
  size_t numel = utils::multiply_integers(tref->sizes);
  api::StagingBuffer staging(
      graph->context(),
      tref->dtype,
      numel,
      vkapi::CopyDirection::HOST_TO_DEVICE);
  graph->update_staging_nbytes_in_cmd(staging.buffer().mem_size_as_size_t());
  size_t nbytes = numel * vkapi::element_size(tref->dtype);

  // In some cases the staging dtype will diverge from the TensorRef dtype. The
  // most common case for this is when the tensor data is float16, but the GPU
  // does not support 16-bit storage buffers. In these cases, the tensor data
  // is manually casted to the staging dtype.
  vkapi::ScalarType staging_dtype = staging.dtype();
  vkapi::ScalarType tref_dtype = tref->dtype;
  if (staging_dtype == tref_dtype) {
    staging.copy_from(tref->data, nbytes);
  } else {
    // Hard-coded type conversion cases
    if (tref_dtype == vkapi::kHalf && staging_dtype == vkapi::kFloat) {
      const uint16_t* casted_data =
          reinterpret_cast<const uint16_t*>(tref->data);
      staging.cast_half_to_float_and_copy_from(casted_data, numel);
    } else if (tref_dtype == vkapi::kLong && staging_dtype == vkapi::kInt) {
      const int64_t* casted_data = reinterpret_cast<const int64_t*>(tref->data);
      staging.cast_and_copy_from<int64_t, int32_t>(casted_data, numel);
    } else if (tref_dtype == vkapi::kDouble && staging_dtype == vkapi::kFloat) {
      const double* casted_data = reinterpret_cast<const double*>(tref->data);
      staging.cast_and_copy_from<double, float>(casted_data, numel);
    } else {
      VK_THROW(
          "Unsupported type conversion from ",
          tref_dtype,
          " to staging dtype ",
          staging_dtype);
    }
  }

  if (--tref->prepack_use_count == 0) {
    tref->free_buffer();
  }

  return staging;
}

void PrepackNode::prepare_pipelines(ComputeGraph* graph) {
  graph->register_pipeline_to_create(
      shader_, local_workgroup_size_, spec_vars_, push_constants_);
}

void PrepackNode::encode(ComputeGraph* graph) {
  api::Context* const context = graph->context();

  context->check_device_capabilities(shader_);

  std::array<uint8_t, kMaxPushConstantSize> push_constants_data;
  uint32_t push_constants_offset = 0;

  for (const auto& push_constant : push_constants_) {
    push_constants_offset += push_constant.write(
        push_constants_data.data(),
        push_constants_offset,
        kMaxPushConstantSize);
  }

  // A buffer packed from keyed data can share its memory with the same packing
  // in other graphs (or other nodes of this one): key it by everything the
  // packing depends on.
  std::string shared_key;
  if (!graph->val_is_none(tref_) && graph->is_buffer_storage(packed_) &&
      !graph->get_tref(tref_)->key.empty()) {
    TensorRefPtr tref = graph->get_tref(tref_);
    shared_key =
        tref->key + "|" + shader_.kernel_name + "|" +
        std::to_string(static_cast<int>(tref->dtype)) + "|" +
        std::to_string(static_cast<int>(graph->dtype_of(packed_))) + "|" +
        std::to_string(graph->hashed_layout_of(packed_)) + "|" +
        std::to_string(
            vTensorPtr(graph, packed_)->get_memory_requirements().size);
    for (const int64_t d : tref->sizes) {
      shared_key += "," + std::to_string(d);
    }
    shared_key += "|";
    for (const int64_t d : graph->sizes_of(packed_)) {
      shared_key += "," + std::to_string(d);
    }
    shared_key += "|";
    for (uint32_t i = 0; i < push_constants_offset; i++) {
      shared_key += std::to_string(push_constants_data[i]) + ".";
    }
    if (std::shared_ptr<vkapi::Allocation> shared =
            SharedConstants::get().find(shared_key)) {
      std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();
      vTensorPtr packed_tensor(graph, packed_);
      packed_tensor->bind_allocation(*shared);
      graph->hold_shared_constant(std::move(shared));
      if (--tref->prepack_use_count == 0) {
        tref->free_buffer();
      }
      // Written by another submission: make it visible to this graph's reads.
      vkapi::PipelineBarrier pipeline_barrier{};
      packed_tensor->buffer(
          pipeline_barrier,
          vkapi::PipelineStage::COMPUTE,
          vkapi::MemoryAccessType::WRITE);
      packed_tensor->buffer(
          pipeline_barrier,
          vkapi::PipelineStage::COMPUTE,
          vkapi::MemoryAccessType::READ);
      context->register_barrier(pipeline_barrier);
      return;
    }
  }

  api::StagingBuffer staging = create_staging_buffer(graph);

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  {
    // If the vTensor is not yet bound to a memory allocation, create a new one
    // and aquire it.
    if (!shared_key.empty()) {
      vkapi::Adapter* const adapter = context->adapter_ptr();
      vTensorPtr packed_tensor(graph, packed_);
      auto allocation =
          std::make_shared<vkapi::Allocation>(adapter->vma().create_allocation(
              packed_tensor->get_memory_requirements(),
              adapter->vma().gpuonly_resource_create_info()));
      packed_tensor->bind_allocation(*allocation);
      SharedConstants::get().insert(shared_key, allocation);
      graph->hold_shared_constant(std::move(allocation));
    } else {
      graph->create_dedicated_allocation_for(packed_);
    }

    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::DescriptorSet descriptor_set = context->get_descriptor_set(
        shader_, local_workgroup_size_, spec_vars_, push_constants_offset);

    uint32_t idx = 0;
    graph->bind_tensor_to_descriptor_set(
        packed_,
        pipeline_barrier,
        vkapi::MemoryAccessType::WRITE,
        descriptor_set,
        idx++);
    bind_staging_to_descriptor_set(staging, descriptor_set, idx++);
    bind_params_to_descriptor_set(params_, descriptor_set, idx);

    context->register_shader_dispatch(
        descriptor_set,
        pipeline_barrier,
        shader_,
        global_workgroup_size_,
        push_constants_data.data(),
        push_constants_offset);
  }

  vkapi::PipelineBarrier pipeline_barrier{};
  vTensorPtr packed_tensor(graph, packed_);
  if (graph->is_buffer_storage(packed_)) {
    packed_tensor->buffer(
        pipeline_barrier,
        vkapi::PipelineStage::COMPUTE,
        vkapi::MemoryAccessType::READ);
  } else {
    packed_tensor->image(
        pipeline_barrier,
        vkapi::PipelineStage::COMPUTE,
        vkapi::MemoryAccessType::READ);
  }
  context->register_barrier(pipeline_barrier);
}

} // namespace vkcompute
