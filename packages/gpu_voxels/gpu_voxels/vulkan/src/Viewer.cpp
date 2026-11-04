// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/Viewer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <optional>

#include <vk_mem_alloc.h>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/PrimitiveArray.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>

#include "Shaders.h"

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kLocalSize = 64;
		constexpr uint32_t kArgumentWords = 5;

		struct BitPush { uint32_t dim_x, dim_y, dim_z, capacity; float side; };
		struct ProbPush { uint32_t dim_x, dim_y, dim_z, capacity; float side; int32_t min_value; };
		struct ListPush { uint32_t count, capacity; float side; };
		struct OctreePush { uint32_t count, capacity; float side; uint32_t probabilistic; int32_t min_value; uint32_t occupied_colour, free_colour; };
		struct PointsPush { uint32_t count, capacity; float side; uint32_t num_clouds, snap; float edge; };
		struct FinalizePush { uint32_t capacity; };

		/// Matches the push constant block of viewer_cube.vert / .frag.
		struct DrawPush
		{
			float view_proj[16];
			uint64_t instances;
			uint32_t shaded;
			uint32_t pad;
			float eye[4];
		};
		static_assert(sizeof(DrawPush) == 96);

		uint32_t group_count(const Context& context, uint64_t items)
		{
			const uint64_t needed = (items + kLocalSize - 1) / kLocalSize;
			const uint64_t cap = std::min<uint64_t>(context.limits().maxComputeWorkGroupCount[0], 65535);
			return static_cast<uint32_t>(std::clamp<uint64_t>(needed, 1, cap));
		}

		void check(VkResult result, const char* what)
		{
			if (result != VK_SUCCESS)
				throw VulkanError(std::string(what) + " failed with VkResult " + std::to_string(static_cast<int>(result)));
		}

		using Mat4 = std::array<float, 16>; // column-major

		Mat4 multiply(const Mat4& a, const Mat4& b)
		{
			Mat4 r{};
			for (int col = 0; col < 4; ++col)
				for (int row = 0; row < 4; ++row)
				{
					float sum = 0.0f;
					for (int k = 0; k < 4; ++k)
						sum += a[k * 4 + row] * b[col * 4 + k];
					r[col * 4 + row] = sum;
				}
			return r;
		}

		std::array<float, 3> cross(const std::array<float, 3>& a, const std::array<float, 3>& b)
		{
			return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
		}

		float dot(const std::array<float, 3>& a, const std::array<float, 3>& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

		std::array<float, 3> normalized(std::array<float, 3> v)
		{
			const float length = std::sqrt(dot(v, v));
			if (!(length > 0.0f))
				throw VulkanError("ViewerCamera: zero length direction (eye == target or up parallel to the view direction)");
			for (float& c : v)
				c /= length;
			return v;
		}

		uint32_t hsv_to_packed(float hue_degrees, float saturation, float value)
		{
			const float h = std::fmod(hue_degrees, 360.0f) / 60.0f;
			const float c = value * saturation;
			const float x = c * (1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f));
			float r = 0, g = 0, b = 0;
			switch (static_cast<int>(h))
			{
			case 0: r = c; g = x; break;
			case 1: r = x; g = c; break;
			case 2: g = c; b = x; break;
			case 3: g = x; b = c; break;
			case 4: r = x; b = c; break;
			default: r = c; b = x; break;
			}
			const float m = value - c;
			const auto byte = [](float v) { return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
			return pack_colour(byte(r + m), byte(g + m), byte(b + m));
		}
	}

	// ---------------------------------------------------------------------------------------------
	BitPalette BitPalette::defaults()
	{
		BitPalette palette;
		for (uint32_t meaning = 1; meaning < 256; ++meaning)
			palette.colour[meaning] = hsv_to_packed(static_cast<float>(meaning) * 137.508f, 0.65f, 0.95f);
		return palette;
	}

	// ---------------------------------------------------------------------------------------------
	CubeBuffer::CubeBuffer(Context& context, uint32_t capacity)
		: capacity_(capacity),
		  instances_(context, static_cast<vk::DeviceSize>(std::max<uint32_t>(capacity, 1)) * kInstanceBytes),
		  arguments_(context, kArgumentWords * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndirectBuffer)
	{
		if (capacity == 0)
			throw VulkanError("CubeBuffer: capacity must be positive");
		arguments_.fill(0);
	}

	uint32_t CubeBuffer::count() const
	{
		return arguments_.download_vector<uint32_t>(kArgumentWords)[1];
	}

	uint32_t CubeBuffer::found() const
	{
		return arguments_.download_vector<uint32_t>(kArgumentWords)[4];
	}

	std::vector<CubeBuffer::Cube> CubeBuffer::download() const
	{
		const uint32_t n = count();
		std::vector<Cube> cubes;
		if (n == 0)
			return cubes;
		const auto words = instances_.download_vector<uint32_t>(static_cast<std::size_t>(n) * (kInstanceBytes / 4));
		for (uint32_t i = 0; i < n; ++i)
		{
			const uint32_t* w = &words[static_cast<std::size_t>(i) * 8];
			Cube cube{};
			std::memcpy(&cube.x, &w[0], 4);
			std::memcpy(&cube.y, &w[1], 4);
			std::memcpy(&cube.z, &w[2], 4);
			std::memcpy(&cube.size, &w[3], 4);
			cube.colour = w[4];
			cubes.push_back(cube);
		}
		return cubes;
	}

	void CubeBuffer::assign(std::span<const Cube> cubes, CubeKind kind)
	{
		kind_ = kind;
		const uint32_t stored = static_cast<uint32_t>(std::min<std::size_t>(cubes.size(), capacity_));
		if (stored > 0)
		{
			std::vector<uint32_t> words(static_cast<std::size_t>(stored) * (kInstanceBytes / 4), 0u);
			for (uint32_t i = 0; i < stored; ++i)
			{
				uint32_t* w = &words[static_cast<std::size_t>(i) * 8];
				std::memcpy(&w[0], &cubes[i].x, 4);
				std::memcpy(&w[1], &cubes[i].y, 4);
				std::memcpy(&w[2], &cubes[i].z, 4);
				std::memcpy(&w[3], &cubes[i].size, 4);
				w[4] = cubes[i].colour;
			}
			instances_.upload(words.data(), words.size() * sizeof(uint32_t));
		}
		const uint32_t arguments[kArgumentWords] = { 36u, stored, 0u, 0u, static_cast<uint32_t>(cubes.size()) };
		arguments_.upload(arguments, sizeof arguments);
	}

	// ---------------------------------------------------------------------------------------------
	struct CubeExtractor::Kernels
	{
		explicit Kernels(Context& context)
			: finalize(context, GVL_SHADER_SPAN(viewer_finalize_comp), 1, sizeof(FinalizePush)),
			  bit(context, GVL_SHADER_SPAN(viewer_extract_bit_comp), 4, sizeof(BitPush)),
			  prob(context, GVL_SHADER_SPAN(viewer_extract_prob_comp), 3, sizeof(ProbPush)),
			  list(context, GVL_SHADER_SPAN(viewer_extract_list_comp), 5, sizeof(ListPush)),
			  octree(context, GVL_SHADER_SPAN(viewer_extract_octree_comp), 3, sizeof(OctreePush)),
			  points(context, GVL_SHADER_SPAN(viewer_extract_points_comp), 5, sizeof(PointsPush))
		{
		}

		ComputePipeline finalize;
		ComputePipeline bit;
		ComputePipeline prob;
		ComputePipeline list;
		ComputePipeline octree;
		ComputePipeline points;
	};

	CubeExtractor::CubeExtractor(Context& context)
		: context_(context), kernels_(std::make_unique<Kernels>(context)), palette_buffer_(context, 256 * sizeof(uint32_t))
	{
	}

	CubeExtractor::~CubeExtractor() = default;

	void CubeExtractor::extract(const BitVoxelMap& map, const BitPalette& palette, CubeBuffer& out)
	{
		out.set_kind(CubeKind::Cube);
		const Dim3ui d = map.dimensions();
		palette_buffer_.upload(palette.colour.data(), sizeof(palette.colour));
		const BitPush push{ d.x, d.y, d.z, out.capacity(), map.voxel_side_length() };
		CommandBatch batch(context_);
		batch.fill(out.arguments(), 0);
		batch.dispatch(kernels_->bit, { &map.voxel_buffer(), &palette_buffer_, &out.instances(), &out.arguments() }, group_count(context_, d.voxel_count()), push);
		batch.barrier();
		batch.dispatch(kernels_->finalize, { &out.arguments() }, 1, FinalizePush{ out.capacity() });
		batch.submit_and_wait();
	}

	void CubeExtractor::extract(const ProbVoxelMap& map, const ProbDisplay& display, CubeBuffer& out)
	{
		out.set_kind(CubeKind::Cube);
		const Dim3ui d = map.dimensions();
		const ProbPush push{ d.x, d.y, d.z, out.capacity(), map.voxel_side_length(), display.min_value };
		CommandBatch batch(context_);
		batch.fill(out.arguments(), 0);
		batch.dispatch(kernels_->prob, { &map.voxel_buffer(), &out.instances(), &out.arguments() }, group_count(context_, d.voxel_count()), push);
		batch.barrier();
		batch.dispatch(kernels_->finalize, { &out.arguments() }, 1, FinalizePush{ out.capacity() });
		batch.submit_and_wait();
	}

	void CubeExtractor::extract(const BitVoxelList& list, const BitPalette& palette, CubeBuffer& out)
	{
		out.set_kind(CubeKind::Cube);
		const uint32_t count = list.size();
		palette_buffer_.upload(palette.colour.data(), sizeof(palette.colour));
		CommandBatch batch(context_);
		batch.fill(out.arguments(), 0);
		if (count > 0)
		{
			const ListPush push{ count, out.capacity(), list.voxel_side_length() };
			batch.dispatch(kernels_->list, { &list.coordinates_buffer(), &list.voxel_buffer(), &palette_buffer_, &out.instances(), &out.arguments() },
				group_count(context_, count), push);
			batch.barrier();
		}
		batch.dispatch(kernels_->finalize, { &out.arguments() }, 1, FinalizePush{ out.capacity() });
		batch.submit_and_wait();
	}

	void CubeExtractor::extract(const Octree& tree, const OctreeDisplay& display, CubeBuffer& out)
	{
		out.set_kind(CubeKind::Cube);
		const float side = display.side_length != 0.0f ? display.side_length : tree.voxel_side_length();
		if (!(side > 0.0f) || !std::isfinite(side))
			throw VulkanError("CubeExtractor: an octree needs a voxel side length (OctreeDisplay::side_length or Octree::set_voxel_side_length)");
		const bool probabilistic = tree.kind() == OctreeKind::Probabilistic;

		OctreeExtractOptions options;
		options.min_level = display.min_level;
		if (!probabilistic)
		{
			// the host selects the statuses that have a visible colour, the kernel gives them the colour (same rule: occupied before free)
			options.status_selection.resize(256);
			for (uint32_t status = 0; status < 256; ++status)
			{
				const uint32_t colour = (status & octree_format::kOccupied) ? display.occupied_colour : (status & octree_format::kFree) ? display.free_colour : 0u;
				options.status_selection[status] = (colour >> 24) != 0 ? 1 : 0;
			}
		}
		const OctreeDeviceCubes cubes = tree.extract_to_device(options);

		CommandBatch batch(context_);
		batch.fill(out.arguments(), 0);
		if (cubes.count > 0)
		{
			const OctreePush push{ cubes.count, out.capacity(), side, probabilistic ? 1u : 0u, display.min_value, display.occupied_colour, display.free_colour };
			batch.dispatch(kernels_->octree, { cubes.records.get(), &out.instances(), &out.arguments() }, group_count(context_, cubes.count), push);
			batch.barrier();
		}
		batch.dispatch(kernels_->finalize, { &out.arguments() }, 1, FinalizePush{ out.capacity() });
		batch.submit_and_wait();
	}

	void CubeExtractor::extract(const MetaPointCloud& clouds, float voxel_side_length, const RobotDisplay& display, CubeBuffer& out)
	{
		if (!(voxel_side_length > 0.0f) || !std::isfinite(voxel_side_length))
			throw VulkanError("CubeExtractor: the voxel side length of a point cloud must be positive and finite");
		if (!(display.point_size > 0.0f) || !std::isfinite(display.point_size))
			throw VulkanError("CubeExtractor: RobotDisplay::point_size must be positive and finite");
		out.set_kind(CubeKind::Cube);
		const uint32_t count = clouds.accumulated_size();
		const uint32_t num_clouds = clouds.num_clouds();
		// the offsets and colours are tiny; they live as long as the batch (submit_and_wait below)
		std::vector<uint32_t> colours(std::max<uint32_t>(num_clouds, 1), 0u);
		const BitPalette hues = BitPalette::defaults();
		for (uint32_t k = 0; k < num_clouds; ++k)
			colours[k] = display.colours.empty() ? hues.colour[1 + k % 255] : display.colours[k % display.colours.size()];
		DeviceBuffer offsets_buffer(context_, std::max<std::size_t>(clouds.offsets().size(), 1) * sizeof(uint32_t));
		DeviceBuffer colours_buffer(context_, colours.size() * sizeof(uint32_t));
		offsets_buffer.upload(clouds.offsets().data(), clouds.offsets().size() * sizeof(uint32_t));
		colours_buffer.upload(colours.data(), colours.size() * sizeof(uint32_t));
		CommandBatch batch(context_);
		batch.fill(out.arguments(), 0);
		if (count > 0 && num_clouds > 0)
		{
			const PointsPush push{ count, out.capacity(), voxel_side_length, num_clouds, display.snap_to_voxels ? 1u : 0u, display.point_size };
			batch.dispatch(kernels_->points, { &clouds.points(), &offsets_buffer, &colours_buffer, &out.instances(), &out.arguments() }, group_count(context_, count), push);
			batch.barrier();
		}
		batch.dispatch(kernels_->finalize, { &out.arguments() }, 1, FinalizePush{ out.capacity() });
		batch.submit_and_wait();
	}

	void CubeExtractor::extract(const PrimitiveArray& array, uint32_t colour, CubeBuffer& out)
	{
		const float side = array.voxel_side_length();
		std::vector<CubeBuffer::Cube> cubes;
		cubes.reserve(array.size());
		for (const PrimitiveEntry& p : array.primitives())
			cubes.push_back({ (p.x - p.radius) * side, (p.y - p.radius) * side, (p.z - p.radius) * side, 2.0f * p.radius * side, colour });
		out.assign(cubes, array.type() == ePRIM_SPHERE ? CubeKind::Sphere : CubeKind::Cube);
	}

	// ---------------------------------------------------------------------------------------------
	std::array<float, 16> ViewerCamera::view_projection(float aspect) const
	{
		const auto forward = normalized({ target[0] - eye[0], target[1] - eye[1], target[2] - eye[2] });
		const auto side = normalized(cross(forward, up));
		const auto true_up = cross(side, forward);
		// view matrix of a right handed look-at (camera looks down -z)
		const Mat4 view{ side[0], true_up[0], -forward[0], 0.0f,
		                 side[1], true_up[1], -forward[1], 0.0f,
		                 side[2], true_up[2], -forward[2], 0.0f,
		                 -dot(side, eye), -dot(true_up, eye), dot(forward, eye), 1.0f };
		// projection for the Vulkan clip space: y points down, depth 0 (near) .. 1 (far)
		const float f = 1.0f / std::tan(fov_y_degrees * std::numbers::pi_v<float> / 360.0f);
		const Mat4 projection{ f / aspect, 0.0f, 0.0f, 0.0f,
		                       0.0f, -f, 0.0f, 0.0f,
		                       0.0f, 0.0f, z_far / (z_near - z_far), -1.0f,
		                       0.0f, 0.0f, (z_near * z_far) / (z_near - z_far), 0.0f };
		return multiply(projection, view);
	}

	// ---------------------------------------------------------------------------------------------
	VoxelRenderer::VoxelRenderer(Context& context, vk::Format colour_format, vk::Format depth_format)
		: context_(context), colour_format_(colour_format), depth_format_(depth_format)
	{
		if (!context.info().graphics)
			throw VulkanError("VoxelRenderer: the Context was not created with ContextCreateInfo::enable_graphics");
		if (!context.info().buffer_device_address)
			throw VulkanError("VoxelRenderer: the device lacks bufferDeviceAddress (the cube instances are read through their address)");

		const vk::Device device = context.device();
		const auto make_module = [&](std::span<const uint32_t> spirv)
		{
			vk::ShaderModuleCreateInfo info{};
			info.setCodeSize(spirv.size_bytes()).setPCode(spirv.data());
			return device.createShaderModuleUnique(info);
		};
		const auto vertex = make_module(GVL_SHADER_SPAN(viewer_cube_vert));
		const auto fragment = make_module(GVL_SHADER_SPAN(viewer_cube_frag));
		const auto sphere_fragment = make_module(GVL_SHADER_SPAN(viewer_sphere_frag));

		const vk::PushConstantRange push_range{ vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPush) };
		vk::PipelineLayoutCreateInfo layout_info{};
		layout_info.setPushConstantRanges(push_range);
		layout_ = device.createPipelineLayoutUnique(layout_info);

		const auto make_pipeline = [&](vk::ShaderModule fragment_module)
		{
			const std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
				vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eVertex, *vertex, "main" },
				vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eFragment, fragment_module, "main" } };
			const vk::PipelineVertexInputStateCreateInfo vertex_input{};
			const vk::PipelineInputAssemblyStateCreateInfo input_assembly{ {}, vk::PrimitiveTopology::eTriangleList, VK_FALSE };
			const vk::PipelineViewportStateCreateInfo viewport_state{ {}, 1, nullptr, 1, nullptr };
			vk::PipelineRasterizationStateCreateInfo raster{};
			raster.setPolygonMode(vk::PolygonMode::eFill).setCullMode(vk::CullModeFlagBits::eNone).setFrontFace(vk::FrontFace::eCounterClockwise).setLineWidth(1.0f);
			const vk::PipelineMultisampleStateCreateInfo multisample{ {}, vk::SampleCountFlagBits::e1 };
			vk::PipelineDepthStencilStateCreateInfo depth_stencil{};
			depth_stencil.setDepthTestEnable(VK_TRUE).setDepthWriteEnable(VK_TRUE).setDepthCompareOp(vk::CompareOp::eLess);
			vk::PipelineColorBlendAttachmentState blend_attachment{};
			blend_attachment.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
			vk::PipelineColorBlendStateCreateInfo blend{};
			blend.setAttachments(blend_attachment);
			const std::array<vk::DynamicState, 2> dynamic_states{ vk::DynamicState::eViewport, vk::DynamicState::eScissor };
			vk::PipelineDynamicStateCreateInfo dynamic{};
			dynamic.setDynamicStates(dynamic_states);

			vk::PipelineRenderingCreateInfo rendering{};
			rendering.setColorAttachmentFormats(colour_format_).setDepthAttachmentFormat(depth_format_);

			vk::GraphicsPipelineCreateInfo pipeline_info{};
			pipeline_info.setStages(stages).setPVertexInputState(&vertex_input).setPInputAssemblyState(&input_assembly).setPViewportState(&viewport_state)
				.setPRasterizationState(&raster).setPMultisampleState(&multisample).setPDepthStencilState(&depth_stencil).setPColorBlendState(&blend)
				.setPDynamicState(&dynamic).setLayout(*layout_).setPNext(&rendering);
			auto result = device.createGraphicsPipelineUnique(nullptr, pipeline_info);
			if (result.result != vk::Result::eSuccess)
				throw VulkanError("createGraphicsPipeline failed");
			return std::move(result.value);
		};
		pipeline_ = make_pipeline(*fragment);
		sphere_pipeline_ = make_pipeline(*sphere_fragment);
	}

	VoxelRenderer::~VoxelRenderer() = default;

	void VoxelRenderer::record(vk::CommandBuffer cb, vk::ImageView colour, vk::ImageView depth, vk::Extent2D extent, const ViewerCamera& camera,
		std::span<const CubeBuffer* const> buffers, const RenderOptions& options) const
	{
		vk::RenderingAttachmentInfo colour_attachment{};
		colour_attachment.setImageView(colour).setImageLayout(vk::ImageLayout::eColorAttachmentOptimal).setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore).setClearValue(vk::ClearValue(vk::ClearColorValue(options.clear_colour)));
		vk::RenderingAttachmentInfo depth_attachment{};
		depth_attachment.setImageView(depth).setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal).setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eDontCare).setClearValue(vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0)));
		vk::RenderingInfo rendering{};
		rendering.setRenderArea(vk::Rect2D({ 0, 0 }, extent)).setLayerCount(1).setColorAttachments(colour_attachment).setPDepthAttachment(&depth_attachment);

		cb.beginRendering(rendering);
		cb.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f));
		cb.setScissor(0, vk::Rect2D({ 0, 0 }, extent));

		DrawPush push{};
		const auto view_projection = camera.view_projection(static_cast<float>(extent.width) / static_cast<float>(std::max(extent.height, 1u)));
		std::memcpy(push.view_proj, view_projection.data(), sizeof push.view_proj);
		push.shaded = options.shaded ? 1u : 0u;
		for (std::size_t i = 0; i < 3; ++i)
			push.eye[i] = camera.eye[i];
		push.eye[3] = 1.0f;
		std::optional<CubeKind> bound;
		for (const CubeBuffer* buffer : buffers)
		{
			if (!bound || *bound != buffer->kind())
			{
				cb.bindPipeline(vk::PipelineBindPoint::eGraphics, buffer->kind() == CubeKind::Sphere ? *sphere_pipeline_ : *pipeline_);
				bound = buffer->kind();
			}
			push.instances = buffer->instances().device_address();
			cb.pushConstants(*layout_, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof push, &push);
			cb.drawIndirect(buffer->arguments().handle(), 0, 1, 4 * sizeof(uint32_t));
		}
		cb.endRendering();
	}

	// ---------------------------------------------------------------------------------------------
	OffscreenTarget::OffscreenTarget(Context& context, uint32_t width, uint32_t height) : context_(context), width_(width), height_(height)
	{
		if (width == 0 || height == 0)
			throw VulkanError("OffscreenTarget: empty size");
		const auto create_image = [&](vk::Format format, vk::ImageUsageFlags usage, vk::Image& image, VmaAllocation_T*& allocation)
		{
			VkImageCreateInfo info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
			info.imageType = VK_IMAGE_TYPE_2D;
			info.format = static_cast<VkFormat>(format);
			info.extent = { width, height, 1 };
			info.mipLevels = 1;
			info.arrayLayers = 1;
			info.samples = VK_SAMPLE_COUNT_1_BIT;
			info.tiling = VK_IMAGE_TILING_OPTIMAL;
			info.usage = static_cast<VkImageUsageFlags>(usage);
			info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			VmaAllocationCreateInfo alloc{};
			alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
			VkImage raw = VK_NULL_HANDLE;
			check(vmaCreateImage(context.allocator(), &info, &alloc, &raw, &allocation, nullptr), "vmaCreateImage");
			image = raw;
		};
		create_image(kColourFormat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc, colour_, colour_allocation_);
		create_image(kDepthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment, depth_, depth_allocation_);

		const auto make_view = [&](vk::Image image, vk::Format format, vk::ImageAspectFlags aspect)
		{
			vk::ImageViewCreateInfo info{};
			info.setImage(image).setViewType(vk::ImageViewType::e2D).setFormat(format).setSubresourceRange({ aspect, 0, 1, 0, 1 });
			return context.device().createImageViewUnique(info);
		};
		colour_view_ = make_view(colour_, kColourFormat, vk::ImageAspectFlagBits::eColor);
		depth_view_ = make_view(depth_, kDepthFormat, vk::ImageAspectFlagBits::eDepth);
		readback_ = std::make_unique<DeviceBuffer>(context, static_cast<vk::DeviceSize>(width) * height * 4);
	}

	OffscreenTarget::~OffscreenTarget()
	{
		colour_view_.reset();
		depth_view_.reset();
		if (colour_allocation_)
			vmaDestroyImage(context_.allocator(), static_cast<VkImage>(colour_), colour_allocation_);
		if (depth_allocation_)
			vmaDestroyImage(context_.allocator(), static_cast<VkImage>(depth_), depth_allocation_);
	}

	void OffscreenTarget::render(const VoxelRenderer& renderer, const ViewerCamera& camera, std::span<const CubeBuffer* const> buffers, const RenderOptions& options)
	{
		if (renderer.colour_format() != kColourFormat || renderer.depth_format() != kDepthFormat)
			throw VulkanError("OffscreenTarget::render: the renderer was created for other formats");
		context_.execute([&](vk::CommandBuffer cb)
		{
			const auto image_barrier = [](vk::Image image, vk::ImageAspectFlags aspect, vk::ImageLayout from, vk::ImageLayout to,
				vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access)
			{
				vk::ImageMemoryBarrier2 barrier{};
				barrier.setSrcStageMask(src_stage).setSrcAccessMask(src_access).setDstStageMask(dst_stage).setDstAccessMask(dst_access)
					.setOldLayout(from).setNewLayout(to).setImage(image).setSubresourceRange({ aspect, 0, 1, 0, 1 })
					.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
				return barrier;
			};
			using Stage = vk::PipelineStageFlagBits2;
			using Access = vk::AccessFlagBits2;
			const std::array<vk::ImageMemoryBarrier2, 2> to_attachment{
				image_barrier(colour_, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
					Stage::eTopOfPipe, {}, Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite),
				image_barrier(depth_, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
					Stage::eTopOfPipe, {}, Stage::eEarlyFragmentTests | Stage::eLateFragmentTests, Access::eDepthStencilAttachmentWrite | Access::eDepthStencilAttachmentRead) };
			cb.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(to_attachment));

			renderer.record(cb, *colour_view_, *depth_view_, vk::Extent2D{ width_, height_ }, camera, buffers, options);

			const auto to_transfer = image_barrier(colour_, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
				vk::ImageLayout::eTransferSrcOptimal, Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite, Stage::eCopy, Access::eTransferRead);
			cb.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(to_transfer));

			vk::BufferImageCopy region{};
			region.setImageSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 }).setImageExtent({ width_, height_, 1 });
			cb.copyImageToBuffer(colour_, vk::ImageLayout::eTransferSrcOptimal, readback_->handle(), region);

			vk::BufferMemoryBarrier2 done{};
			done.setSrcStageMask(Stage::eCopy).setSrcAccessMask(Access::eTransferWrite).setDstStageMask(Stage::eCopy).setDstAccessMask(Access::eTransferRead)
				.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).setBuffer(readback_->handle()).setOffset(0).setSize(VK_WHOLE_SIZE);
			cb.pipelineBarrier2(vk::DependencyInfo{}.setBufferMemoryBarriers(done));
		});
	}

	std::vector<uint8_t> OffscreenTarget::read_rgba() const
	{
		return readback_->download_vector<uint8_t>(static_cast<std::size_t>(width_) * height_ * 4);
	}
}
