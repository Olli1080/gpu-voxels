// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/DeviceBuffer.h>

#include <algorithm>
#include <utility>

#include <vk_mem_alloc.h>

namespace gpu_voxels::vulkan
{
	namespace
	{
		/// Transfers larger than this are split so that the staging buffer stays small.
		constexpr vk::DeviceSize kStagingChunkBytes = 64ull * 1024 * 1024;

		void check(VkResult result, const char* what)
		{
			if (result != VK_SUCCESS)
				throw VulkanError(std::string(what) + " failed with VkResult " + std::to_string(static_cast<int>(result)));
		}

		/// Host-visible staging buffer, destroyed on scope exit.
		class StagingBuffer
		{
		public:
			StagingBuffer(Context& context, vk::DeviceSize bytes, bool for_readback) : allocator_(context.allocator())
			{
				VkBufferCreateInfo info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
				info.size = bytes;
				info.usage = for_readback ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
				info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
				VmaAllocationCreateInfo alloc{};
				alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
				alloc.flags = for_readback ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
				                           : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
				check(vmaCreateBuffer(allocator_, &info, &alloc, &buffer_, &allocation_, nullptr), "vmaCreateBuffer(staging)");
			}
			~StagingBuffer() { vmaDestroyBuffer(allocator_, buffer_, allocation_); }
			StagingBuffer(const StagingBuffer&) = delete;
			StagingBuffer& operator=(const StagingBuffer&) = delete;

			[[nodiscard]] vk::Buffer handle() const { return buffer_; }
			void write(const void* data, vk::DeviceSize bytes)
			{
				check(vmaCopyMemoryToAllocation(allocator_, data, allocation_, 0, bytes), "vmaCopyMemoryToAllocation");
			}
			void read(void* data, vk::DeviceSize bytes)
			{
				check(vmaCopyAllocationToMemory(allocator_, allocation_, 0, data, bytes), "vmaCopyAllocationToMemory");
			}

		private:
			VmaAllocator allocator_;
			VkBuffer buffer_ = VK_NULL_HANDLE;
			VmaAllocation allocation_ = nullptr;
		};
	}

	DeviceBuffer::DeviceBuffer(Context& context, vk::DeviceSize size_bytes, vk::BufferUsageFlags extra_usage)
		: context_(&context), size_(size_bytes)
	{
		if (size_bytes == 0)
			throw VulkanError("DeviceBuffer: zero-sized buffers are not allowed");
		if (size_bytes > context.limits().maxStorageBufferRange)
		{
			// A buffer is always bound as a whole by CommandBatch; larger buffers need range binding (not implemented).
			throw VulkanError("DeviceBuffer: " + std::to_string(size_bytes) + " bytes exceed maxStorageBufferRange (" +
				std::to_string(context.limits().maxStorageBufferRange) + ")");
		}

		VkBufferCreateInfo info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		info.size = size_bytes;
		vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
			vk::BufferUsageFlagBits::eTransferDst | extra_usage;
		if (context.info().buffer_device_address)
			usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
		info.usage = static_cast<VkBufferUsageFlags>(usage);
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VmaAllocationCreateInfo alloc{};
		alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = nullptr;
		check(vmaCreateBuffer(context.allocator(), &info, &alloc, &buffer, &allocation, nullptr), "vmaCreateBuffer");
		buffer_ = buffer;
		allocation_ = allocation;
		if (context.info().buffer_device_address)
			address_ = context.device().getBufferAddress(vk::BufferDeviceAddressInfo{ buffer_ });
	}

	vk::DeviceAddress DeviceBuffer::device_address() const
	{
		if (address_ == 0)
			throw VulkanError("DeviceBuffer::device_address: the device does not support bufferDeviceAddress");
		return address_;
	}

	DeviceBuffer::~DeviceBuffer() { release(); }

	void DeviceBuffer::release() noexcept
	{
		if (context_ && allocation_)
			vmaDestroyBuffer(context_->allocator(), static_cast<VkBuffer>(buffer_), allocation_);
		context_ = nullptr;
		buffer_ = nullptr;
		allocation_ = nullptr;
		size_ = 0;
	}

	DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
		: context_(std::exchange(other.context_, nullptr)), buffer_(std::exchange(other.buffer_, nullptr)),
		  allocation_(std::exchange(other.allocation_, nullptr)), size_(std::exchange(other.size_, 0)),
		  address_(std::exchange(other.address_, 0))
	{
	}

	DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept
	{
		if (this != &other)
		{
			release();
			context_ = std::exchange(other.context_, nullptr);
			buffer_ = std::exchange(other.buffer_, nullptr);
			allocation_ = std::exchange(other.allocation_, nullptr);
			size_ = std::exchange(other.size_, 0);
			address_ = std::exchange(other.address_, 0);
		}
		return *this;
	}

	void DeviceBuffer::upload(const void* data, vk::DeviceSize bytes, vk::DeviceSize offset)
	{
		if (offset > size_ || bytes > size_ - offset)
			throw VulkanError("DeviceBuffer::upload out of range");
		const auto* src = static_cast<const std::byte*>(data);
		for (vk::DeviceSize done = 0; done < bytes;)
		{
			const vk::DeviceSize chunk = std::min(kStagingChunkBytes, bytes - done);
			StagingBuffer staging(*context_, chunk, false);
			staging.write(src + done, chunk);
			const vk::BufferCopy region{ 0, offset + done, chunk };
			context_->execute([&](vk::CommandBuffer cb) { cb.copyBuffer(staging.handle(), buffer_, region); });
			done += chunk;
		}
	}

	void DeviceBuffer::download(void* data, vk::DeviceSize bytes, vk::DeviceSize offset) const
	{
		if (offset > size_ || bytes > size_ - offset)
			throw VulkanError("DeviceBuffer::download out of range");
		auto* dst = static_cast<std::byte*>(data);
		for (vk::DeviceSize done = 0; done < bytes;)
		{
			const vk::DeviceSize chunk = std::min(kStagingChunkBytes, bytes - done);
			StagingBuffer staging(*context_, chunk, true);
			const vk::BufferCopy region{ offset + done, 0, chunk };
			context_->execute([&](vk::CommandBuffer cb) { cb.copyBuffer(buffer_, staging.handle(), region); });
			staging.read(dst + done, chunk);
			done += chunk;
		}
	}

	void DeviceBuffer::fill(uint32_t value)
	{
		if (size_ % 4 != 0)
			throw VulkanError("DeviceBuffer::fill requires a size that is a multiple of 4");
		context_->execute([&](vk::CommandBuffer cb) { cb.fillBuffer(buffer_, 0, VK_WHOLE_SIZE, value); });
	}
}
