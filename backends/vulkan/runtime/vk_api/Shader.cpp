/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/Shader.h>

#include <cstring>
#include <utility>
#include <vector>

namespace vkcompute {
namespace vkapi {

//
// ShaderInfo
//

ShaderInfo::ShaderInfo()
    : src_code{
          nullptr,
          0u,
      } {}

ShaderInfo::ShaderInfo(
    std::string name,
    const uint32_t* const spirv_bin,
    const uint32_t size,
    std::vector<VkDescriptorType>  layout,
    const utils::uvec3 tile_size,
    const bool requires_shader_int16_ext,
    const bool requires_16bit_storage_ext,
    const bool requires_8bit_storage_ext,
    const bool requires_integer_dot_product_ext,
    const bool requires_shader_int64_ext,
    const bool requires_shader_float64_ext,
    const uint32_t required_subgroup_size_arg)
    : src_code{
          spirv_bin,
          size,
      },
      kernel_name{std::move(name)},
      kernel_layout{std::move(layout)},
      out_tile_size(tile_size),
      requires_shader_int16(requires_shader_int16_ext),
      requires_16bit_storage(requires_16bit_storage_ext),
      requires_8bit_storage(requires_8bit_storage_ext),
      requires_integer_dot_product(requires_integer_dot_product_ext),
      requires_shader_int64(requires_shader_int64_ext),
      requires_shader_float64(requires_shader_float64_ext),
      required_subgroup_size(required_subgroup_size_arg) {
}

bool operator==(const ShaderInfo& _1, const ShaderInfo& _2) {
  return (
      _1.src_code.bin == _2.src_code.bin &&
      _1.src_code.size == _2.src_code.size);
}

namespace {

constexpr uint32_t kOpExtension = 10u;
constexpr uint32_t kOpEntryPoint = 15u;
constexpr uint32_t kOpExecutionMode = 16u;
constexpr uint32_t kOpCapability = 17u;
constexpr uint32_t kOpTypeFloat = 22u;
constexpr uint32_t kCapabilityRoundingModeRTE = 4467u;
constexpr uint32_t kExecutionModeRoundingModeRTE = 4462u;

uint32_t instruction(const uint32_t word_count, const uint32_t opcode) {
  return (word_count << 16u) | opcode;
}

// The code with RoundingModeRTE for 16 bits added to the entry point, or empty
// if the module declares no 16-bit float or already sets a rounding mode.
std::vector<uint32_t> add_fp16_round_to_nearest(
    const uint32_t* code,
    const size_t count) {
  if (count < 5u || code[0] != 0x07230203u) {
    return {};
  }
  size_t capabilities_end = 5u;
  size_t entry_points_end = 0u;
  uint32_t entry_point = 0u;
  bool has_fp16 = false;
  for (size_t i = 5u; i < count;) {
    const uint32_t opcode = code[i] & 0xFFFFu;
    const uint32_t word_count = code[i] >> 16u;
    if (word_count == 0u || i + word_count > count) {
      return {};
    }
    if (opcode == kOpCapability) {
      if (code[i + 1] == kCapabilityRoundingModeRTE) {
        return {};
      }
      capabilities_end = i + word_count;
    } else if (opcode == kOpEntryPoint) {
      if (entry_point == 0u) {
        entry_point = code[i + 2];
      }
      entry_points_end = i + word_count;
    } else if (opcode == kOpTypeFloat && code[i + 2] == 16u) {
      has_fp16 = true;
    }
    i += word_count;
  }
  if (!has_fp16 || entry_points_end == 0u) {
    return {};
  }

  std::vector<uint32_t> out(code, code + capabilities_end);
  out.push_back(instruction(2u, kOpCapability));
  out.push_back(kCapabilityRoundingModeRTE);
  // Float controls are core from SPIR-V 1.4.
  if (code[1] < 0x00010400u) {
    const char name[] = "SPV_KHR_float_controls";
    std::vector<uint32_t> literal((sizeof(name) + 3u) / 4u, 0u);
    std::memcpy(literal.data(), name, sizeof(name));
    out.push_back(
        instruction(1u + static_cast<uint32_t>(literal.size()), kOpExtension));
    out.insert(out.end(), literal.begin(), literal.end());
  }
  out.insert(out.end(), code + capabilities_end, code + entry_points_end);
  out.push_back(instruction(4u, kOpExecutionMode));
  out.push_back(entry_point);
  out.push_back(kExecutionModeRoundingModeRTE);
  out.push_back(16u);
  out.insert(out.end(), code + entry_points_end, code + count);
  return out;
}

} // namespace

ShaderInfo with_fp16_round_to_nearest(const ShaderInfo& shader) {
  static std::mutex mutex;
  static std::unordered_map<const uint32_t*, std::vector<uint32_t>> patched;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = patched.find(shader.src_code.bin);
  if (it == patched.end()) {
    it = patched
             .emplace(
                 shader.src_code.bin,
                 add_fp16_round_to_nearest(
                     shader.src_code.bin, shader.src_code.size / 4u))
             .first;
  }
  if (it->second.empty()) {
    return shader;
  }
  ShaderInfo out = shader;
  out.src_code.bin = it->second.data();
  out.src_code.size = static_cast<uint32_t>(it->second.size() * 4u);
  return out;
}

//
// ShaderLayout
//

ShaderLayout::ShaderLayout(
    VkDevice device,
    const ShaderLayout::Signature& signature)
    : device_(device), handle_{VK_NULL_HANDLE} {
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(signature.size());

  uint32_t binding_num = 0u;
  for (const VkDescriptorType type : signature) {
    bindings.emplace_back(VkDescriptorSetLayoutBinding{
        binding_num++, // binding
        type, // descriptorType
        1u, // descriptorCount
        VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
        nullptr, // pImmutableSamplers
    });
  }

  const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(bindings.size()), // bindingCount
      bindings.data(), // pBindings
  };

  VK_CHECK(vkCreateDescriptorSetLayout(
      device_, &descriptor_set_layout_create_info, nullptr, &handle_));
}

ShaderLayout::ShaderLayout(ShaderLayout&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ShaderLayout::~ShaderLayout() {
  if (handle_ == VK_NULL_HANDLE) {
    return;
  }
  vkDestroyDescriptorSetLayout(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkDescriptorSetLayout tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderModule
//

ShaderModule::ShaderModule(VkDevice device, const ShaderInfo& source)
    : device_(device), handle_{VK_NULL_HANDLE} {
  const uint32_t* code = source.src_code.bin;
  uint32_t size = source.src_code.size;

  const VkShaderModuleCreateInfo shader_module_create_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // codeSize
      code, // pCode
  };

  VK_CHECK(vkCreateShaderModule(
      device_, &shader_module_create_info, nullptr, &handle_));
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ShaderModule::~ShaderModule() {
  if (handle_ == VK_NULL_HANDLE) {
    return;
  }
  vkDestroyShaderModule(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkShaderModule tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderLayoutCache
//

ShaderLayoutCache::ShaderLayoutCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

ShaderLayoutCache::ShaderLayoutCache(ShaderLayoutCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

ShaderLayoutCache::~ShaderLayoutCache() {
  purge();
}

VkDescriptorSetLayout ShaderLayoutCache::retrieve(
    const ShaderLayoutCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, ShaderLayoutCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void ShaderLayoutCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// ShaderCache
//

ShaderCache::ShaderCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

ShaderCache::ShaderCache(ShaderCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

ShaderCache::~ShaderCache() {
  purge();
}

VkShaderModule ShaderCache::retrieve(const ShaderCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, ShaderCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void ShaderCache::purge() {
  cache_.clear();
}

} // namespace vkapi
} // namespace vkcompute
