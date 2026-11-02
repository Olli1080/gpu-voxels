// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Disk format of voxel maps and voxel lists (slice S7-4), counterpart of the CUDA writeToDisk / readFromDisk.
 *
 * The Vulkan native format, little endian, one file per map or list:
 *   header (48 bytes): magic "GVLMAP\0\1" (8), version u32 (1), MapType u32, dimensions x y z u32 (maps: their size, lists: the reference map),
 *   voxel side length f32, reserved u32 (0), reserved u32 (0), count u64 (voxels of a map, entries of a list);
 *   payload: bit map: count * 8 words (the layout of BitVoxelMap::download_words()); prob. map: count int8 occupancies; distance map: count
 *   packed words (x | y << 10 | z << 20); bit list: ids (u32 * count), coordinates (3 * u32 * count), bit vectors (8 * u32 * count);
 *   counting list: ids, coordinates, counts (int8 * count);
 *   trailer: FNV-1a (64 bit) over header and payload, u64.
 * The CUDA formats are raw dumps of the device memory (different voxel layouts for the prob. and distance map) and are not read here: nothing
 * could be tested against files written by CUDA. A read replaces the whole content, or changes nothing if the file is refused. Octrees have
 * their own format (OctreeNativeStorage).
 */
#ifndef GPU_VOXELS_VULKAN_MAP_FILE_H_INCLUDED
#define GPU_VOXELS_VULKAN_MAP_FILE_H_INCLUDED

#include <filesystem>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/DistanceVoxelMap.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>
#include <gpu_voxels/vulkan/vulkan_common.h>

namespace gpu_voxels::vulkan
{
	/// A file that cannot be opened, written, or read as the map it is given to: unknown format, a type, size or voxel side length that does not
	/// match the target, wrong length, a wrong checksum, entries that are no valid list. Derived from VulkanError; GpuVoxelsMap::read_from_disk /
	/// write_to_disk turn exactly this one into `false` (CUDA returns false and logs).
	class MapFileError : public VulkanError
	{
	public:
		using VulkanError::VulkanError;
	};

	/// \throws MapFileError if the file cannot be written
	void write_map_file(const std::filesystem::path& file, const BitVoxelMap& map);
	void write_map_file(const std::filesystem::path& file, const ProbVoxelMap& map);
	void write_map_file(const std::filesystem::path& file, const DistanceVoxelMap& map);
	void write_map_file(const std::filesystem::path& file, const BitVoxelList& list);
	void write_map_file(const std::filesystem::path& file, const CountingVoxelList& list);

	/// Replaces the content of `map` / `list` by the file. The type, the dimensions (lists: the reference dimensions) and the voxel side length of the
	/// file must equal those of the target (CUDA warns about a different side length of a map and continues, lists refuse it).
	/// \throws MapFileError for every reason above; the target is unchanged then
	void read_map_file(const std::filesystem::path& file, BitVoxelMap& map);
	void read_map_file(const std::filesystem::path& file, ProbVoxelMap& map);
	void read_map_file(const std::filesystem::path& file, DistanceVoxelMap& map);
	void read_map_file(const std::filesystem::path& file, BitVoxelList& list);
	void read_map_file(const std::filesystem::path& file, CountingVoxelList& list);
}

#endif
