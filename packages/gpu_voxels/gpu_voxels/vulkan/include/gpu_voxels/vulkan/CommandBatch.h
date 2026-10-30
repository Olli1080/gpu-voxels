// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Records several compute dispatches and barriers into ONE command buffer and submits it once.
 */
#ifndef GPU_VOXELS_VULKAN_COMMAND_BATCH_H_INCLUDED
#define GPU_VOXELS_VULKAN_COMMAND_BATCH_H_INCLUDED

#include <cstdint>
#include <initializer_list>
#include <span>
#include <type_traits>
#include <vector>

#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>

namespace gpu_voxels::vulkan
{
	/*!
	 * Typical use:
	 * \code
	 *   CommandBatch batch(ctx);
	 *   batch.dispatch(add_one, {&data}, groups, 1, 1, &push, sizeof(push));
	 *   batch.barrier();                      // later dispatches see the writes of earlier ones
	 *   batch.dispatch(scale, {&data, &out}, groups);
	 *   batch.submit_and_wait();              // one submit for everything
	 * \endcode
	 * Dispatches are NOT ordered with respect to each other unless barrier() is called in between.
	 * Pipelines and buffers must stay alive until submit_and_wait() returned. A batch is single-use.
	 */
	class CommandBatch
	{
	public:
		explicit CommandBatch(Context& context);
		CommandBatch(const CommandBatch&) = delete;
		CommandBatch& operator=(const CommandBatch&) = delete;

		/// \param buffers must contain exactly pipeline.storage_buffer_count() buffers (binding i = buffers[i])
		/// \param push    push constant data of exactly pipeline.push_constant_bytes() bytes (nullptr if none)
		void dispatch(const ComputePipeline& pipeline, std::span<const DeviceBuffer* const> buffers,
			uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1,
			const void* push = nullptr, uint32_t push_bytes = 0);

		void dispatch(const ComputePipeline& pipeline, std::initializer_list<const DeviceBuffer*> buffers,
			uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1,
			const void* push = nullptr, uint32_t push_bytes = 0)
		{
			dispatch(pipeline, std::span<const DeviceBuffer* const>(buffers.begin(), buffers.size()),
				groups_x, groups_y, groups_z, push, push_bytes);
		}

		/// Typed push constants.
		template<typename Push>
		void dispatch(const ComputePipeline& pipeline, std::initializer_list<const DeviceBuffer*> buffers,
			uint32_t groups_x, const Push& push)
		{
			static_assert(std::is_trivially_copyable_v<Push>);
			dispatch(pipeline, buffers, groups_x, 1, 1, &push, static_cast<uint32_t>(sizeof(Push)));
		}

		/// Memory dependency: storage-buffer writes of all earlier dispatches become visible to later dispatches.
		void barrier();

		/// Fills the whole buffer with a 32-bit pattern (size multiple of 4). Includes the barrier that makes the
		/// fill visible to all later dispatches of this batch.
		void fill(DeviceBuffer& buffer, uint32_t value);

		/// Copies `bytes` between two different buffers. Includes full memory barriers before and after the copy, so
		/// it is ordered with all earlier and later commands of the batch.
		void copy(const DeviceBuffer& source, DeviceBuffer& destination, vk::DeviceSize bytes,
			vk::DeviceSize source_offset = 0, vk::DeviceSize destination_offset = 0);

		/// Ends recording, submits and waits. \throws VulkanError if called twice.
		void submit_and_wait();

		[[nodiscard]] uint32_t dispatch_count() const { return dispatch_count_; }

	private:
		vk::DescriptorSet allocate_descriptor_set(vk::DescriptorSetLayout layout);

		Context& context_;
		vk::UniqueCommandPool pool_;
		vk::CommandBuffer command_buffer_;
		std::vector<vk::UniqueDescriptorPool> descriptor_pools_;
		uint32_t sets_in_current_pool_ = 0;
		uint32_t dispatch_count_ = 0;
		bool submitted_ = false;
	};
}

#endif
