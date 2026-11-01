// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Storing and restoring an Octree (slice S6e): the interface and the native Vulkan format.
 *
 * CUDA NTree::serialize writes the cubes of an extraction (text or binary) and deserialize rebuilds a tree from them. Here the tree is
 * reached through an interface, so that a reader for the CUDA format can be added behind it without touching the tree (decision Q22):
 * OctreeStorage works on Octree::dump() / Octree::load_nodes() only. OctreeNativeStorage is the first implementation.
 */
#ifndef GPU_VOXELS_VULKAN_OCTREE_STORAGE_H_INCLUDED
#define GPU_VOXELS_VULKAN_OCTREE_STORAGE_H_INCLUDED

#include <cstdint>
#include <iosfwd>
#include <string>

#include <gpu_voxels/vulkan/Octree.h>

namespace gpu_voxels::vulkan
{
	class OctreeStorage
	{
	public:
		virtual ~OctreeStorage() = default;

		/// Writes the tree. Pending updates are propagated first (the stored statuses must be consistent), so `tree` is not const.
		/// \throws VulkanError if the stream fails
		virtual void write(Octree& tree, std::ostream& out) const = 0;
		/// Replaces the tree by the stored one (kind and voxel side length are checked / restored, flags are clear, voxel_count() is 0).
		/// \throws VulkanError for data that is not a valid tree of the kind of `tree`; `tree` is unchanged then
		virtual void read(Octree& tree, std::istream& in) const = 0;

		/// CUDA writeToDisk / readFromDisk (they return false and log): the file variants throw VulkanError when the file cannot be opened,
		/// written or read.
		void write_file(Octree& tree, const std::string& path) const;
		void read_file(Octree& tree, const std::string& path) const;
	};

	/// The Vulkan native format, little endian:
	///   header (32 bytes): magic "GVLOCT\0\1" (8), version u32 (1), kind u32 (0 deterministic, 1 probabilistic), voxel side length f32 (metres,
	///   0 = not set), reserved u32 (0), node count u64;
	///   node records (16 bytes each, ordered as Octree::dump()): id u64, level u8, status u8, occupancy i8, has_children u8 (0 / 1), reserved u32 (0);
	///   trailer: FNV-1a (64 bit) over the header and all records, u64.
	/// It stores exactly the nodes reachable from the root (cut subtrees, i.e. garbage, are not stored), with status, occupancy and collision bits,
	/// so a restored tree is equal to the original in dump(), extract(), find() and collisions. Where CUDA's text / binary dump holds the cubes of
	/// an extraction restricted by the status selection, this one holds the whole tree.
	class OctreeNativeStorage final : public OctreeStorage
	{
	public:
		static constexpr uint32_t kVersion = 1;
		static constexpr uint32_t kHeaderBytes = 32;
		static constexpr uint32_t kRecordBytes = 16;
		static constexpr uint32_t kTrailerBytes = 8;

		void write(Octree& tree, std::ostream& out) const override;
		void read(Octree& tree, std::istream& in) const override;
	};
}

#endif
