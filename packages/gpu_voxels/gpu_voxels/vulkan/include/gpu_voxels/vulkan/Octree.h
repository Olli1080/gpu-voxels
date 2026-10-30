// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Octree on the Vulkan backend, pointer-faithful port of the CUDA NTree (slice S6b: build, find, extract, check_tree).
 *
 * 15 levels (0 = voxels, 14 = root), branching factor 8, Morton ids (voxel coordinates < 2^14 per axis). Every level is one
 * chunk (buffer) of blocks of 8 sibling nodes; an inner node holds the 48-bit device address of its child block, so the tree
 * needs the optional feature bufferDeviceAddress (DeviceInfo::buffer_device_address); the constructor throws without it.
 * Node formats: OctreeFormat.h. Vulkan gives no help for wrong node addresses (Agent/knowledge_base/vulkan_migration/
 * octree_porting_hazards.md): check_tree() validates every pointer on the host and is meant to be called after every mutation
 * in tests.
 *
 * build() computes the final node states directly (bottom-up while the level blocks are created); this is the tree CUDA's
 * build + propagate leaves behind. Insertion, propagation and rebuild of an existing tree come with slice S6d.
 */
#ifndef GPU_VOXELS_VULKAN_OCTREE_H_INCLUDED
#define GPU_VOXELS_VULKAN_OCTREE_H_INCLUDED

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/OctreeFormat.h>

namespace gpu_voxels::vulkan
{
	struct OctreeVoxel
	{
		uint32_t x = 0, y = 0, z = 0;
	};

	/// A node of the memory tree.
	struct OctreeNode
	{
		uint32_t level = 0;
		uint64_t id = 0;       ///< Morton prefix: voxel id >> 3 * level
		uint8_t status = 0;
		int8_t occupancy = 0;  ///< probabilistic tree only
		bool has_children = false;
		friend bool operator==(const OctreeNode&, const OctreeNode&) = default;
	};

	/// Result of find(): the node where the lookup stopped.
	struct OctreeLookup
	{
		static constexpr uint32_t kOutside = 255;
		uint32_t level = 0;  ///< kOutside when the id is >= 2^42
		uint8_t status = 0;
		int8_t occupancy = 0;
		friend bool operator==(const OctreeLookup&, const OctreeLookup&) = default;
	};

	/// A node emitted by extract().
	struct OctreeCube
	{
		uint32_t level = 0;
		uint64_t id = 0;       ///< Morton prefix: voxel id >> 3 * level
		uint8_t status = 0;
		int8_t occupancy = 0;  ///< probabilistic tree only
		friend bool operator==(const OctreeCube&, const OctreeCube&) = default;
	};

	struct OctreeExtractOptions
	{
		/// Nodes of this level are emitted even when they are refined further (0..13; the root is never emitted).
		uint32_t min_level = 0;
		/// 256 flags indexed by the full status byte; empty = extract every node.
		std::vector<uint8_t> status_selection;
	};

	struct OctreeStats
	{
		uint64_t bytes = 0;                  ///< all chunks
		uint64_t blocks[octree_format::kLevelCount] = {}; ///< allocated blocks per level (a block holds 8 nodes)
		uint64_t nodes() const { uint64_t n = 0; for (auto b : blocks) n += 8 * b; return n; }
	};

	class Octree
	{
	public:
		static constexpr uint32_t kLevels = octree_format::kLevelCount;
		static constexpr uint32_t kCoordinateLimit = 1u << 14;
		/// Maximum number of input voxels of one build (limit of the scan / sort primitives).
		static constexpr uint32_t kMaxVoxels = 1u << 24;

		/// \throws VulkanError if the device lacks bufferDeviceAddress. Starts as an empty tree (the root only).
		Octree(Context& context, OctreeKind kind);
		~Octree();
		Octree(Octree&&) noexcept;
		Octree& operator=(Octree&&) noexcept;
		Octree(const Octree&) = delete;
		Octree& operator=(const Octree&) = delete;

		[[nodiscard]] OctreeKind kind() const;
		[[nodiscard]] Context& context() const;

		/// Replaces the tree by the tree of the given voxels (duplicates are merged). Voxels with a coordinate >= 2^14 are dropped
		/// and counted in dropped_count() (CUDA only asserts). Blocking.
		void build(std::span<const OctreeVoxel> voxels);
		/// Same for Morton ids (voxel coordinates interleaved, see Morton.h); ids >= 2^42 are dropped.
		void build_from_ids(std::span<const uint64_t> morton_ids);
		/// Back to the empty tree; frees all chunks.
		void clear();

		/// Distinct voxels of the last build.
		[[nodiscard]] uint32_t voxel_count() const;
		/// Input voxels of the last build that were outside the tree.
		[[nodiscard]] uint64_t dropped_count() const;
		[[nodiscard]] OctreeStats memory_usage() const;

		/// Looks up every id on the GPU: the walk from the root stops at a leaf, a node without children or a uniform node
		/// (no PART flag).
		[[nodiscard]] std::vector<OctreeLookup> find(std::span<const uint64_t> morton_ids) const;
		/// Cubes of the tree on the GPU (rules of the CUDA extract kernel); the order of the result is unspecified.
		[[nodiscard]] std::vector<OctreeCube> extract(const OctreeExtractOptions& options = {}) const;

		/// All nodes reachable from the root, ordered by (level, id); read back to the host and walked there.
		[[nodiscard]] std::vector<OctreeNode> dump() const;
		/// Raw access to the chunk of a level (blocks of 8 nodes, formats in OctreeFormat.h) for tests and tools; writing to it can
		/// corrupt the tree (check_tree() finds that). 	hrows VulkanError if the level has no chunk.
		[[nodiscard]] DeviceBuffer& chunk(uint32_t level);
		/// Test hook: caps the x group count of every dispatch (default 65535) so that the 2D dispatch path is exercised with small trees.
		void set_max_groups_x(uint32_t groups);

		/// Validates the structure on the host (pointers inside registered chunks, alignment, every block referenced once,
		/// status rules). \throws VulkanError listing the first problems.
		void check_tree() const;

	private:
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};
}

#endif
