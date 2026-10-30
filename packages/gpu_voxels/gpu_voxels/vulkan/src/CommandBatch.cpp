// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/CommandBatch.h>

#include <array>

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kSetsPerPool = 64;
	}

	CommandBatch::CommandBatch(Context& context) : context_(context)
	{
		vk::CommandPoolCreateInfo pool_info{};
		pool_info.setQueueFamilyIndex(context.queue_family()).setFlags(vk::CommandPoolCreateFlagBits::eTransient);
		pool_ = context.device().createCommandPoolUnique(pool_info);

		vk::CommandBufferAllocateInfo alloc_info{};
		alloc_info.setCommandPool(*pool_).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1);
		command_buffer_ = context.device().allocateCommandBuffers(alloc_info).front();

		vk::CommandBufferBeginInfo begin_info{};
		begin_info.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		command_buffer_.begin(begin_info);
	}

	vk::DescriptorSet CommandBatch::allocate_descriptor_set(vk::DescriptorSetLayout layout)
	{
		if (descriptor_pools_.empty() || sets_in_current_pool_ >= kSetsPerPool)
		{
			const vk::DescriptorPoolSize size{ vk::DescriptorType::eStorageBuffer,
				kSetsPerPool * ComputePipeline::kMaxStorageBuffers };
			vk::DescriptorPoolCreateInfo info{};
			info.setMaxSets(kSetsPerPool).setPoolSizes(size);
			descriptor_pools_.push_back(context_.device().createDescriptorPoolUnique(info));
			sets_in_current_pool_ = 0;
		}
		vk::DescriptorSetAllocateInfo alloc{};
		alloc.setDescriptorPool(*descriptor_pools_.back()).setSetLayouts(layout);
		++sets_in_current_pool_;
		return context_.device().allocateDescriptorSets(alloc).front();
	}

	void CommandBatch::dispatch(const ComputePipeline& pipeline, std::span<const DeviceBuffer* const> buffers,
		uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, const void* push, uint32_t push_bytes)
	{
		if (submitted_)
			throw VulkanError("CommandBatch: already submitted");
		if (buffers.size() != pipeline.storage_buffer_count())
			throw VulkanError("CommandBatch::dispatch: expected " + std::to_string(pipeline.storage_buffer_count()) +
				" buffers, got " + std::to_string(buffers.size()));
		if (push_bytes != pipeline.push_constant_bytes() || (push_bytes > 0 && push == nullptr))
			throw VulkanError("CommandBatch::dispatch: push constant size mismatch");
		const auto& max_groups = context_.limits().maxComputeWorkGroupCount;
		if (groups_x > max_groups[0] || groups_y > max_groups[1] || groups_z > max_groups[2])
			throw VulkanError("CommandBatch::dispatch: group count exceeds maxComputeWorkGroupCount");
		for (const DeviceBuffer* b : buffers)
			if (b == nullptr)
				throw VulkanError("CommandBatch::dispatch: null buffer");

		std::array<vk::DescriptorBufferInfo, ComputePipeline::kMaxStorageBuffers> buffer_infos{};
		std::array<vk::WriteDescriptorSet, ComputePipeline::kMaxStorageBuffers> writes{};
		const vk::DescriptorSet set = allocate_descriptor_set(pipeline.descriptor_set_layout());
		for (uint32_t i = 0; i < buffers.size(); ++i)
		{
			buffer_infos[i] = vk::DescriptorBufferInfo{ buffers[i]->handle(), 0, VK_WHOLE_SIZE };
			writes[i].setDstSet(set).setDstBinding(i).setDescriptorCount(1)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer).setPBufferInfo(&buffer_infos[i]);
		}
		if (!buffers.empty())
			context_.device().updateDescriptorSets(static_cast<uint32_t>(buffers.size()), writes.data(), 0, nullptr);

		command_buffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline());
		command_buffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.layout(), 0, set, {});
		if (push_bytes > 0)
			command_buffer_.pushConstants(pipeline.layout(), vk::ShaderStageFlagBits::eCompute, 0, push_bytes, push);
		command_buffer_.dispatch(groups_x, groups_y, groups_z);
		++dispatch_count_;
	}

	void CommandBatch::barrier()
	{
		if (submitted_)
			throw VulkanError("CommandBatch: already submitted");
		vk::MemoryBarrier2 memory{};
		memory.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
			.setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
			.setDstAccessMask(vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
		vk::DependencyInfo dependency{};
		dependency.setMemoryBarriers(memory);
		command_buffer_.pipelineBarrier2(dependency);
	}

	void CommandBatch::fill(DeviceBuffer& buffer, uint32_t value)
	{
		if (submitted_)
			throw VulkanError("CommandBatch: already submitted");
		if (buffer.size() % 4 != 0)
			throw VulkanError("CommandBatch::fill requires a size that is a multiple of 4");
		command_buffer_.fillBuffer(buffer.handle(), 0, VK_WHOLE_SIZE, value);
		vk::MemoryBarrier2 memory{};
		memory.setSrcStageMask(vk::PipelineStageFlagBits2::eAllTransfer)
			.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
			.setDstAccessMask(vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
		vk::DependencyInfo dependency{};
		dependency.setMemoryBarriers(memory);
		command_buffer_.pipelineBarrier2(dependency);
	}

	void CommandBatch::copy(const DeviceBuffer& source, DeviceBuffer& destination, vk::DeviceSize bytes,
		vk::DeviceSize source_offset, vk::DeviceSize destination_offset)
	{
		if (submitted_)
			throw VulkanError("CommandBatch: already submitted");
		if (&source == &destination)
			throw VulkanError("CommandBatch::copy: source and destination must differ");
		if (source_offset > source.size() || bytes > source.size() - source_offset ||
			destination_offset > destination.size() || bytes > destination.size() - destination_offset)
			throw VulkanError("CommandBatch::copy: range out of bounds");
		if (bytes == 0)
			return;

		auto barrier_between = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
			vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access)
		{
			vk::MemoryBarrier2 memory{};
			memory.setSrcStageMask(src_stage).setSrcAccessMask(src_access).setDstStageMask(dst_stage).setDstAccessMask(dst_access);
			vk::DependencyInfo dependency{};
			dependency.setMemoryBarriers(memory);
			command_buffer_.pipelineBarrier2(dependency);
		};
		barrier_between(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
			vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
		const vk::BufferCopy region{ source_offset, destination_offset, bytes };
		command_buffer_.copyBuffer(source.handle(), destination.handle(), region);
		barrier_between(vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	}

	void CommandBatch::submit_and_wait()
	{
		if (submitted_)
			throw VulkanError("CommandBatch: already submitted");
		submitted_ = true;
		command_buffer_.end();
		context_.submit_and_wait(command_buffer_);
	}
}
