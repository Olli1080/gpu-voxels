// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Device-local storage buffer with explicit upload / download (no host mirror, no unified memory).
 */
#ifndef GPU_VOXELS_VULKAN_DEVICE_BUFFER_H_INCLUDED
#define GPU_VOXELS_VULKAN_DEVICE_BUFFER_H_INCLUDED

#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include <gpu_voxels/vulkan/Context.h>

struct VmaAllocation_T;

namespace gpu_voxels::vulkan
{
	/*!
	 * Device-local buffer usable as storage buffer and as transfer source/destination.
	 * Host access happens only through upload()/download() (staging copies, blocking).
	 * The Context must outlive the buffer.
	 */
	class DeviceBuffer
	{
	public:
		/// \param size_bytes  must be > 0 and <= the device's maxBufferSize
		/// \param extra_usage additional usage flags (e.g. vk::BufferUsageFlagBits::eIndirectBuffer)
		DeviceBuffer(Context& context, vk::DeviceSize size_bytes, vk::BufferUsageFlags extra_usage = {});
		~DeviceBuffer();
		DeviceBuffer(DeviceBuffer&& other) noexcept;
		DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
		DeviceBuffer(const DeviceBuffer&) = delete;
		DeviceBuffer& operator=(const DeviceBuffer&) = delete;

		[[nodiscard]] vk::Buffer handle() const { return buffer_; }
		[[nodiscard]] vk::DeviceSize size() const { return size_; }
		/// Address of the first byte for shaders (GL_EXT_buffer_reference). Valid while the buffer lives; stays valid when the
		/// buffer object is moved. \throws VulkanError if the device lacks bufferDeviceAddress (DeviceInfo::buffer_device_address).
		[[nodiscard]] vk::DeviceAddress device_address() const;

		/// Copy host data into the buffer at byte offset. Blocks until finished.
		void upload(const void* data, vk::DeviceSize bytes, vk::DeviceSize offset = 0);
		/// Copy buffer contents to host memory. Blocks until finished.
		void download(void* data, vk::DeviceSize bytes, vk::DeviceSize offset = 0) const;
		/// Fill the whole buffer with a 32-bit pattern (size must be a multiple of 4).
		void fill(uint32_t value);

		template<typename T>
		void upload(std::span<const T> values, vk::DeviceSize offset_bytes = 0)
		{
			static_assert(std::is_trivially_copyable_v<T>);
			upload(values.data(), values.size_bytes(), offset_bytes);
		}

		template<typename T>
		[[nodiscard]] std::vector<T> download_vector(std::size_t count, vk::DeviceSize offset_bytes = 0) const
		{
			static_assert(std::is_trivially_copyable_v<T>);
			std::vector<T> out(count);
			download(out.data(), count * sizeof(T), offset_bytes);
			return out;
		}

	private:
		void release() noexcept;

		Context* context_ = nullptr;
		vk::Buffer buffer_;
		VmaAllocation_T* allocation_ = nullptr;
		vk::DeviceSize size_ = 0;
		vk::DeviceAddress address_ = 0;
	};
}

#endif
