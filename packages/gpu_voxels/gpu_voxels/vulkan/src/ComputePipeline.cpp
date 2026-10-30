// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/ComputePipeline.h>

#include <fstream>

namespace gpu_voxels::vulkan
{
	ComputePipeline::ComputePipeline(Context& context, std::span<const uint32_t> spirv, uint32_t storage_buffer_count,
		uint32_t push_constant_bytes)
		: storage_buffer_count_(storage_buffer_count), push_constant_bytes_(push_constant_bytes)
	{
		if (spirv.empty())
			throw VulkanError("ComputePipeline: empty SPIR-V");
		if (storage_buffer_count > kMaxStorageBuffers)
			throw VulkanError("ComputePipeline: more than " + std::to_string(kMaxStorageBuffers) + " storage buffers");
		if (push_constant_bytes % 4 != 0 || push_constant_bytes > context.limits().maxPushConstantsSize)
			throw VulkanError("ComputePipeline: invalid push constant size " + std::to_string(push_constant_bytes));

		const vk::Device device = context.device();

		std::vector<vk::DescriptorSetLayoutBinding> bindings;
		for (uint32_t i = 0; i < storage_buffer_count; ++i)
		{
			vk::DescriptorSetLayoutBinding b{};
			b.setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute);
			bindings.push_back(b);
		}
		vk::DescriptorSetLayoutCreateInfo set_info{};
		set_info.setBindings(bindings);
		set_layout_ = device.createDescriptorSetLayoutUnique(set_info);

		vk::PushConstantRange push_range{ vk::ShaderStageFlagBits::eCompute, 0, push_constant_bytes };
		vk::PipelineLayoutCreateInfo layout_info{};
		layout_info.setSetLayouts(*set_layout_);
		if (push_constant_bytes > 0)
			layout_info.setPushConstantRanges(push_range);
		layout_ = device.createPipelineLayoutUnique(layout_info);

		vk::ShaderModuleCreateInfo module_info{};
		module_info.setCodeSize(spirv.size_bytes()).setPCode(spirv.data());
		shader_ = device.createShaderModuleUnique(module_info);

		vk::PipelineShaderStageCreateInfo stage{};
		stage.setStage(vk::ShaderStageFlagBits::eCompute).setModule(*shader_).setPName("main");
		vk::ComputePipelineCreateInfo pipeline_info{};
		pipeline_info.setStage(stage).setLayout(*layout_);
		auto result = device.createComputePipelineUnique(nullptr, pipeline_info);
		if (result.result != vk::Result::eSuccess)
			throw VulkanError("createComputePipeline failed");
		pipeline_ = std::move(result.value);
	}

	std::vector<uint32_t> ComputePipeline::load_spirv(const std::filesystem::path& file)
	{
		std::ifstream in(file, std::ios::binary | std::ios::ate);
		if (!in)
			throw VulkanError("Cannot open SPIR-V file " + file.string());
		const auto size = static_cast<std::size_t>(in.tellg());
		if (size == 0 || size % 4 != 0)
			throw VulkanError("Invalid SPIR-V file size: " + file.string());
		std::vector<uint32_t> words(size / 4);
		in.seekg(0);
		in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
		if (!in)
			throw VulkanError("Cannot read SPIR-V file " + file.string());
		return words;
	}
}
