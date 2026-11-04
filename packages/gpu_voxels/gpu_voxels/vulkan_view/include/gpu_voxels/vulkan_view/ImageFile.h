// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Minimal PNG writer for screenshots of the viewer (RGBA, 8 bit, stored deflate blocks: no compression, no dependency).
 */
#ifndef GPU_VOXELS_VULKAN_IMAGE_FILE_H_INCLUDED
#define GPU_VOXELS_VULKAN_IMAGE_FILE_H_INCLUDED

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include <gpu_voxels/vulkan/vulkan_common.h>

namespace gpu_voxels::vulkan
{
	/// Encode `rgba` (width * height * 4 bytes, rows from the top) as a PNG file image.
	/// \throws VulkanError if the size does not match
	[[nodiscard]] std::vector<uint8_t> encode_png(uint32_t width, uint32_t height, std::span<const uint8_t> rgba);
	/// Write a PNG file. \throws VulkanError if the file cannot be written
	void write_png(const std::filesystem::path& file, uint32_t width, uint32_t height, std::span<const uint8_t> rgba);
}

#endif
