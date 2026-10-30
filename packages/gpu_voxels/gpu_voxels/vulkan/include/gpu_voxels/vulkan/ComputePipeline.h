// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Compute pipeline for a SPIR-V kernel with N storage buffers (set 0, bindings 0..N-1) and push constants.
 */
#ifndef GPU_VOXELS_VULKAN_COMPUTE_PIPELINE_H_INCLUDED
#define GPU_VOXELS_VULKAN_COMPUTE_PIPELINE_H_INCLUDED

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include <gpu_voxels/vulkan/Context.h>

namespace gpu_voxels::vulkan
{
	class ComputePipeline
	{
	public:
		static constexpr uint32_t kMaxStorageBuffers = 16;

		/*!
		 * \param spirv                 SPIR-V words of a compute shader (entry point "main")
		 * \param storage_buffer_count  number of storage buffers at set 0, binding 0..count-1 (<= kMaxStorageBuffers)
		 * \param push_constant_bytes   size of the push constant block (0 = none, multiple of 4, <= maxPushConstantsSize)
		 */
		ComputePipeline(Context& context, std::span<const uint32_t> spirv, uint32_t storage_buffer_count,
			uint32_t push_constant_bytes = 0);

		/// Read a .spv file. \throws VulkanError if it cannot be read or its size is not a multiple of 4.
		static std::vector<uint32_t> load_spirv(const std::filesystem::path& file);

		[[nodiscard]] vk::Pipeline pipeline() const { return *pipeline_; }
		[[nodiscard]] vk::PipelineLayout layout() const { return *layout_; }
		[[nodiscard]] vk::DescriptorSetLayout descriptor_set_layout() const { return *set_layout_; }
		[[nodiscard]] uint32_t storage_buffer_count() const { return storage_buffer_count_; }
		[[nodiscard]] uint32_t push_constant_bytes() const { return push_constant_bytes_; }

	private:
		vk::UniqueDescriptorSetLayout set_layout_;
		vk::UniquePipelineLayout layout_;
		vk::UniqueShaderModule shader_;
		vk::UniquePipeline pipeline_;
		uint32_t storage_buffer_count_ = 0;
		uint32_t push_constant_bytes_ = 0;
	};
}

#endif
