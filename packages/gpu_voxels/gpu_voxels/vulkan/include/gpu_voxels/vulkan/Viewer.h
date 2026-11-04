// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  In-process voxel viewer core of the Vulkan backend: cubes are extracted from the maps ON THE GPU into an instance buffer and drawn with
 *         one indirect draw per buffer. No second process, no shared memory, no Boost.Interprocess, no host copy of the voxels (the viewer of the
 *         CUDA version shared device pointers with a separate process through CUDA IPC handles; see Agent/knowledge_base/architecture/visualization_and_ipc.md).
 *
 * Pieces:
 *   CubeExtractor  compute kernels: BitVoxelMap / ProbVoxelMap / bit VoxelList / Octree -> CubeBuffer (a rewrite replaces the previous content)
 *   VoxelRenderer  graphics pipeline (dynamic rendering, no render pass objects) that draws CubeBuffers into any colour + depth attachment
 *   OffscreenTarget  colour + depth image with a read back of the pixels (tests, screenshots, headless use)
 * The window / swapchain part is ViewerWindow.h.
 *
 * Needs a Context created with ContextCreateInfo::enable_graphics = true (graphics + compute queue, dynamic rendering) and bufferDeviceAddress.
 */
#ifndef GPU_VOXELS_VULKAN_VIEWER_H_INCLUDED
#define GPU_VOXELS_VULKAN_VIEWER_H_INCLUDED

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

struct VmaAllocation_T;

namespace gpu_voxels::vulkan
{
	class BitVoxelList;
	class BitVoxelMap;
	class Octree;
	class ProbVoxelMap;

	/// Packs a colour as the viewer shaders read it (red in the lowest byte, alpha in the highest).
	constexpr uint32_t pack_colour(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
	{
		return static_cast<uint32_t>(r) | (static_cast<uint32_t>(g) << 8) | (static_cast<uint32_t>(b) << 16) | (static_cast<uint32_t>(a) << 24);
	}

	/// Colour of every meaning (bit index) of a bit map or bit list. A voxel is drawn in the colour of its lowest meaning with a visible colour;
	/// alpha 0 hides a meaning (a voxel with only hidden meanings is not drawn).
	struct BitPalette
	{
		std::array<uint32_t, 256> colour{};

		/// Meaning 0 (eBVM_FREE) hidden, every other meaning a distinct, fully opaque colour (golden-angle hues).
		static BitPalette defaults();
		void set(uint32_t meaning, uint32_t packed_colour) { colour.at(meaning) = packed_colour; }
		void hide(uint32_t meaning) { colour.at(meaning) = 0; }
	};

	struct ProbDisplay
	{
		/// Voxels with an occupancy (int8 log-odds, -127 .. 127) of at least this value are drawn; unknown voxels (-128) never. The colour is a
		/// gradient from blue (-127) over green to red (+127).
		int min_value = 1;
	};

	/// What is drawn of an octree. A node is a cube of 2^level voxels per side at its Morton position; uniform nodes (no children) are drawn as one big
	/// cube, so a mostly empty or mostly full tree needs few cubes.
	struct OctreeDisplay
	{
		/// Nodes on this level or above are not refined further: a refined node is drawn as one cube that stands for its whole subtree (0..13; 0 = every
		/// voxel as exact as the tree stores it). Deterministic tree: such a cube is occupied if any voxel in it is, else free if any is; probabilistic
		/// tree: it has the largest occupancy below it.
		uint32_t min_level = 0;
		/// Edge of a voxel in metres; 0 = Octree::voxel_side_length() (which must then be set).
		float side_length = 0.0f;
		/// Deterministic tree: colour (pack_colour) of occupied cubes and of free ones; alpha 0 hides them (free space is hidden by default).
		uint32_t occupied_colour = pack_colour(235, 140, 40);
		uint32_t free_colour = 0;
		/// Probabilistic tree: nodes with a known occupancy of at least this value are drawn in the gradient colour (as ProbDisplay).
		int min_value = 1;
	};

	/// A list of cubes on the GPU (instance records of 32 bytes) and the arguments of the indirect draw that draws them.
	class CubeBuffer
	{
	public:
		static constexpr uint32_t kInstanceBytes = 32;
		/// \param capacity maximum number of cubes; more are counted but not stored (found() - count() = dropped)
		CubeBuffer(Context& context, uint32_t capacity);

		[[nodiscard]] uint32_t capacity() const { return capacity_; }
		/// Number of cubes that will be drawn (downloads the arguments: blocks).
		[[nodiscard]] uint32_t count() const;
		/// Number of cubes the last extraction found, also those beyond the capacity.
		[[nodiscard]] uint32_t found() const;
		[[nodiscard]] uint32_t dropped() const { return found() - count(); }
		/// The stored cubes as (x, y, z, size, packed colour) -- blocks; for tests and tools.
		struct Cube
		{
			float x, y, z, size;
			uint32_t colour;
		};
		[[nodiscard]] std::vector<Cube> download() const;

		[[nodiscard]] const DeviceBuffer& instances() const { return instances_; }
		[[nodiscard]] const DeviceBuffer& arguments() const { return arguments_; }
		DeviceBuffer& instances() { return instances_; }
		DeviceBuffer& arguments() { return arguments_; }

	private:
		uint32_t capacity_;
		DeviceBuffer instances_;
		DeviceBuffer arguments_; ///< VkDrawIndirectCommand (4 words) + the number found
	};

	class CubeExtractor
	{
	public:
		explicit CubeExtractor(Context& context);
		~CubeExtractor();
		CubeExtractor(const CubeExtractor&) = delete;
		CubeExtractor& operator=(const CubeExtractor&) = delete;

		/// Replace the content of `out` by the cubes of the map / list (one blocking submission). The cube of voxel c is [c, c + 1) * side.
		void extract(const BitVoxelMap& map, const BitPalette& palette, CubeBuffer& out);
		void extract(const ProbVoxelMap& map, const ProbDisplay& display, CubeBuffer& out);
		void extract(const BitVoxelList& list, const BitPalette& palette, CubeBuffer& out);
		/// Same for an octree: the nodes are extracted by the octree kernels into a temporary device buffer (Octree::extract_to_device, a few submissions
		/// with small read backs of counters, no read back of the nodes) and turned into instances. Propagate pending inserts first (statuses are only
		/// consistent after Octree::propagate()). \throws VulkanError if neither the display nor the tree has a side length, for min_level > 13.
		void extract(const Octree& tree, const OctreeDisplay& display, CubeBuffer& out);

	private:
		struct Kernels;
		Context& context_;
		std::unique_ptr<Kernels> kernels_;
		DeviceBuffer palette_buffer_;
	};

	struct ViewerCamera
	{
		std::array<float, 3> eye{ 0.0f, 0.0f, 5.0f };
		std::array<float, 3> target{ 0.0f, 0.0f, 0.0f };
		std::array<float, 3> up{ 0.0f, 1.0f, 0.0f };
		float fov_y_degrees = 60.0f;
		float z_near = 0.01f;
		float z_far = 10000.0f;

		/// Column-major projection * view for the Vulkan clip space (y down, depth 0 .. 1) and the aspect ratio width / height.
		[[nodiscard]] std::array<float, 16> view_projection(float aspect) const;
	};

	struct RenderOptions
	{
		std::array<float, 4> clear_colour{ 0.08f, 0.09f, 0.11f, 1.0f };
		/// true: fixed directional light (the faces of a cube get different shades); false: the exact palette colour (tests).
		bool shaded = true;
	};

	class VoxelRenderer
	{
	public:
		/// \throws VulkanError if the context was not created with enable_graphics or lacks bufferDeviceAddress
		VoxelRenderer(Context& context, vk::Format colour_format, vk::Format depth_format = vk::Format::eD32Sfloat);
		~VoxelRenderer();
		VoxelRenderer(const VoxelRenderer&) = delete;
		VoxelRenderer& operator=(const VoxelRenderer&) = delete;

		[[nodiscard]] vk::Format colour_format() const { return colour_format_; }
		[[nodiscard]] vk::Format depth_format() const { return depth_format_; }

		/// Record the drawing into `cb`: begin rendering (clears colour and depth), one indirect draw per buffer, end rendering. The colour image must
		/// be in the layout eColorAttachmentOptimal and the depth image in eDepthAttachmentOptimal; the instance and argument buffers must have been
		/// written by commands that are complete or ordered before (a CubeExtractor call waits for its submission).
		void record(vk::CommandBuffer cb, vk::ImageView colour, vk::ImageView depth, vk::Extent2D extent, const ViewerCamera& camera,
			std::span<const CubeBuffer* const> buffers, const RenderOptions& options = {}) const;

	private:
		Context& context_;
		vk::Format colour_format_;
		vk::Format depth_format_;
		vk::UniquePipelineLayout layout_;
		vk::UniquePipeline pipeline_;
	};

	/// Colour + depth image of a fixed size (R8G8B8A8_UNORM, D32_SFLOAT) to render into without a window.
	class OffscreenTarget
	{
	public:
		OffscreenTarget(Context& context, uint32_t width, uint32_t height);
		~OffscreenTarget();
		OffscreenTarget(const OffscreenTarget&) = delete;
		OffscreenTarget& operator=(const OffscreenTarget&) = delete;

		[[nodiscard]] uint32_t width() const { return width_; }
		[[nodiscard]] uint32_t height() const { return height_; }
		static constexpr vk::Format kColourFormat = vk::Format::eR8G8B8A8Unorm;
		static constexpr vk::Format kDepthFormat = vk::Format::eD32Sfloat;

		/// Render and wait. `renderer` must have been created with kColourFormat / kDepthFormat.
		void render(const VoxelRenderer& renderer, const ViewerCamera& camera, std::span<const CubeBuffer* const> buffers, const RenderOptions& options = {});
		/// The pixels of the last render() as RGBA bytes, row by row from the top (width * height * 4 bytes).
		[[nodiscard]] std::vector<uint8_t> read_rgba() const;

	private:
		Context& context_;
		uint32_t width_, height_;
		vk::Image colour_ = nullptr, depth_ = nullptr;
		VmaAllocation_T* colour_allocation_ = nullptr;
		VmaAllocation_T* depth_allocation_ = nullptr;
		vk::UniqueImageView colour_view_, depth_view_;
		std::unique_ptr<DeviceBuffer> readback_;
	};
}

#endif
