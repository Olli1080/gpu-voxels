// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Vulkan instance + device + compute queue + allocator (RAII).
 */
#ifndef GPU_VOXELS_VULKAN_CONTEXT_H_INCLUDED
#define GPU_VOXELS_VULKAN_CONTEXT_H_INCLUDED

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/vulkan_common.h>

struct VmaAllocator_T; // vk_mem_alloc.h is not part of the public interface

namespace gpu_voxels::vulkan
{
	struct ContextCreateInfo
	{
		std::string application_name = "gpu_voxels";
		/// Enable VK_LAYER_KHRONOS_validation (incl. synchronization validation) when it is installed.
		bool enable_validation = true;
		/// Throw ValidationUnavailableError instead of continuing without validation.
		bool require_validation = false;
		/// Physical device index; -1 = best device. Overridden by the environment variable GVL_VULKAN_DEVICE.
		int device_index = -1;
		/// Use bufferDeviceAddress when the device has it. false switches it off (tests of the fallback path, or to rule it out when debugging).
		bool enable_buffer_device_address = true;
	};

	/// Properties of the selected device that matter for the voxel kernels.
	struct DeviceInfo
	{
		std::string name;
		std::string type;                  ///< "discrete", "integrated", "virtual", "cpu", "other"
		uint32_t vendor_id = 0;
		uint32_t device_id = 0;
		uint32_t api_version = 0;
		uint32_t driver_version = 0;
		uint32_t queue_family = 0;
		uint32_t subgroup_size = 0;
		uint32_t min_subgroup_size = 0;    ///< 0 if VK_EXT_subgroup_size_control data is unavailable
		uint32_t max_subgroup_size = 0;
		uint64_t device_local_heap_bytes = 0;
		bool validation_enabled = false;
		bool shader_int64_buffer_atomics = false; ///< optional feature, not enabled
		/// Optional feature bufferDeviceAddress (core 1.2): enabled whenever the device has it. Needed by the octree (nodes
		/// hold device addresses of child blocks); DeviceBuffer::device_address() throws without it.
		bool buffer_device_address = false;
	};

	struct ValidationState; ///< implementation detail (validation message counters)

	class Context
	{
	public:
		/// \throws NoDeviceError, ValidationUnavailableError, VulkanError, vk::SystemError
		explicit Context(const ContextCreateInfo& create_info = {});
		~Context();
		Context(const Context&) = delete;
		Context& operator=(const Context&) = delete;

		[[nodiscard]] vk::Instance instance() const { return *instance_; }
		[[nodiscard]] vk::PhysicalDevice physical_device() const { return physical_device_; }
		[[nodiscard]] vk::Device device() const { return *device_; }
		[[nodiscard]] vk::Queue queue() const { return queue_; }
		[[nodiscard]] uint32_t queue_family() const { return info_.queue_family; }
		[[nodiscard]] VmaAllocator_T* allocator() const { return allocator_; }
		[[nodiscard]] const DeviceInfo& info() const { return info_; }
		[[nodiscard]] const vk::PhysicalDeviceLimits& limits() const { return limits_; }

		/// Record into a one-time command buffer, submit it and wait for completion.
		void execute(const std::function<void(vk::CommandBuffer)>& record);
		/// Submit an already recorded primary command buffer and wait for completion.
		void submit_and_wait(vk::CommandBuffer command_buffer);
		/// Wait until the device is idle (serialised with submissions).
		void wait_idle();

		// ---- validation results (only meaningful if info().validation_enabled) ----
		[[nodiscard]] uint32_t validation_error_count() const;
		[[nodiscard]] uint32_t validation_warning_count() const;
		/// Copies of up to 64 recorded warning/error messages.
		[[nodiscard]] std::vector<std::string> validation_messages() const;
		void reset_validation_state();

	private:
		vk::UniqueInstance instance_;
		std::unique_ptr<ValidationState> validation_;
		vk::PhysicalDevice physical_device_;
		vk::UniqueDevice device_;
		vk::Queue queue_;
		VmaAllocator_T* allocator_ = nullptr;
		DeviceInfo info_;
		vk::PhysicalDeviceLimits limits_;
		std::mutex queue_mutex_; ///< vkQueueSubmit / vkQueueWaitIdle need external synchronisation
	};
}

#endif
