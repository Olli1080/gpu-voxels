// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Common includes and error types of the Vulkan backend runtime.
 *
 * The runtime uses vulkan.hpp with the statically linked loader (vulkan-1) and exceptions.
 * Required device capabilities (policy Q4, Agent/project_state/open_questions.md):
 * Vulkan 1.3 core, synchronization2 and shaderInt64. Everything else is optional.
 */
#ifndef GPU_VOXELS_VULKAN_COMMON_H_INCLUDED
#define GPU_VOXELS_VULKAN_COMMON_H_INCLUDED

#include <stdexcept>
#include <string>

#include <vulkan/vulkan.hpp>

namespace gpu_voxels::vulkan
{
	/// Base class of all errors raised by the Vulkan backend runtime (besides vk::SystemError from vulkan.hpp).
	class VulkanError : public std::runtime_error
	{
	public:
		using std::runtime_error::runtime_error;
	};

	/// No Vulkan device satisfies the requirements.
	class NoDeviceError : public VulkanError
	{
	public:
		using VulkanError::VulkanError;
	};

	/// Validation was required but VK_LAYER_KHRONOS_validation is not available (install the Vulkan SDK).
	class ValidationUnavailableError : public VulkanError
	{
	public:
		using VulkanError::VulkanError;
	};
}

#endif
