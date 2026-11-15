/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/containers/SharedObject.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

bool SharedObject::has_user(const ValueRef idx) const {
  return std::find(users.begin(), users.end(), idx) != users.end();
}

void SharedObject::add_user(ComputeGraph* const graph, const ValueRef idx) {
  vTensorPtr t = graph->get_tensor(idx);

  // Aggregate Memory Requirements
  const VkMemoryRequirements mem_reqs = t->get_memory_requirements();
  aggregate_memory_requirements.size =
      std::max(mem_reqs.size, aggregate_memory_requirements.size);
  aggregate_memory_requirements.alignment =
      std::max(mem_reqs.alignment, aggregate_memory_requirements.alignment);
  aggregate_memory_requirements.memoryTypeBits |= mem_reqs.memoryTypeBits;

  users.emplace_back(idx);
}

void SharedObject::allocate(ComputeGraph* const graph) {
  if (aggregate_memory_requirements.size == 0) {
    return;
  }

  VmaAllocationCreateInfo alloc_create_info =
      graph->context()->adapter_ptr()->vma().gpuonly_resource_create_info();

  allocation = graph->context()->adapter_ptr()->vma().create_allocation(
      aggregate_memory_requirements, alloc_create_info);
}

SharedWorkspace& SharedWorkspace::get() {
  static SharedWorkspace workspace;
  return workspace;
}

void SharedWorkspace::reserve(
    const void* graph,
    const Memory memory,
    const VkMemoryRequirements& requirements) {
  std::lock_guard<std::mutex> lock(mutex_);
  pools_[memory].reserved[graph] = requirements;
}

void SharedWorkspace::release(const void* graph) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (Pool& pool : pools_) {
    pool.reserved.erase(graph);
    if (pool.reserved.empty()) {
      // Graphs still bound to it keep it alive.
      pool.block.reset();
    }
  }
}

std::shared_ptr<vkapi::Allocation> SharedWorkspace::acquire(
    vkapi::Adapter* adapter,
    const Memory memory,
    const VkMemoryRequirements& requirements) {
  std::lock_guard<std::mutex> lock(mutex_);
  Pool& pool = pools_[memory];
  const bool fits = pool.block && requirements.size <= pool.size &&
      pool.alignment % requirements.alignment == 0 &&
      (requirements.memoryTypeBits & (1u << pool.memory_type)) != 0;
  if (!fits) {
    VkMemoryRequirements total = requirements;
    for (const auto& entry : pool.reserved) {
      total.size = std::max(total.size, entry.second.size);
      total.alignment = std::max(total.alignment, entry.second.alignment);
      total.memoryTypeBits &= entry.second.memoryTypeBits;
    }
    VK_CHECK_COND(
        total.memoryTypeBits != 0,
        "No memory type suits every graph of the shared workspace");
    VmaAllocationCreateInfo create_info =
        adapter->vma().gpuonly_resource_create_info();
    if (memory == kHost) {
      // Staging in both directions: mapped, and cached for the reads back.
      create_info = {};
      create_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
      create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
      create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
          VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
    pool.block = std::make_shared<vkapi::Allocation>(
        adapter->vma().create_allocation(total, create_info));
    VmaAllocationInfo info;
    vmaGetAllocationInfo(pool.block->allocator, pool.block->allocation, &info);
    pool.size = total.size;
    pool.alignment = total.alignment;
    pool.memory_type = info.memoryType;
  }
  return pool.block;
}

SharedConstants& SharedConstants::get() {
  static SharedConstants constants;
  return constants;
}

std::shared_ptr<vkapi::Allocation> SharedConstants::find(
    const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return nullptr;
  }
  std::shared_ptr<vkapi::Allocation> allocation = it->second.lock();
  if (!allocation) {
    allocations_.erase(it);
  }
  return allocation;
}

void SharedConstants::insert(
    const std::string& key,
    const std::shared_ptr<vkapi::Allocation>& allocation) {
  std::lock_guard<std::mutex> lock(mutex_);
  allocations_[key] = allocation;
}

void SharedObject::bind_users(ComputeGraph* const graph) {
  if (users.empty()) {
    return;
  }
  for (const ValueRef idx : users) {
    graph->get_tensor(idx)->bind_allocation(allocation);
  }
}

} // namespace vkcompute
