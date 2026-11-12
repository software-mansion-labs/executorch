/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/vk_api/Types.h>

#include <executorch/backends/vulkan/runtime/graph/GraphConfig.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vkcompute {

class ComputeGraph;

struct SharedObject {
  friend class ComputeGraph;

  explicit SharedObject() = default;

  VkMemoryRequirements aggregate_memory_requirements;
  std::vector<ValueRef> users;
  vkapi::Allocation allocation;

  bool has_user(const ValueRef idx) const;
  void add_user(ComputeGraph* const graph, const ValueRef idx);
  void allocate(ComputeGraph* const graph);
  void bind_users(ComputeGraph* const graph);
};

/*
 * Working memory for graphs whose executions never overlap (see
 * GraphConfig::shared_workspace): one device allocation for the shared objects
 * and one mapped host allocation for the staging buffers, each sized for the
 * largest graph that has reserved space, in place of each graph's own. Graphs
 * bind at their first execution, so graphs loaded together are all counted
 * before the allocations are made; a graph that does not fit later gets new,
 * larger ones.
 */
class SharedWorkspace final {
 public:
  enum Memory { kDevice = 0, kHost = 1 };

  static SharedWorkspace& get();

  // Held by a graph from copying its inputs in to copying its outputs out.
  std::mutex& execute_mutex() {
    return execute_mutex_;
  }

  void reserve(
      const void* graph,
      Memory memory,
      const VkMemoryRequirements& requirements);
  void release(const void* graph);
  std::shared_ptr<vkapi::Allocation> acquire(
      vkapi::Adapter* adapter,
      Memory memory,
      const VkMemoryRequirements& requirements);

 private:
  struct Pool {
    std::unordered_map<const void*, VkMemoryRequirements> reserved;
    std::shared_ptr<vkapi::Allocation> block;
    VkDeviceSize size = 0;
    VkDeviceSize alignment = 0;
    uint32_t memory_type = 0;
  };

  std::mutex mutex_;
  std::mutex execute_mutex_;
  Pool pools_[2];
};

/*
 * Packed constants shared between graphs: a graph that would pack the same
 * keyed data with the same shader into an identical buffer binds the existing
 * allocation instead. Entries live as long as some graph holds them.
 */
class SharedConstants final {
 public:
  static SharedConstants& get();

  std::shared_ptr<vkapi::Allocation> find(const std::string& key);
  void insert(
      const std::string& key,
      const std::shared_ptr<vkapi::Allocation>& allocation);

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, std::weak_ptr<vkapi::Allocation>>
      allocations_;
};

} // namespace vkcompute
