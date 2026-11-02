// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Host point cloud readers (.xyz, .pcd, .binvox) and the box of voxels for insertBoxIntoMap (slice S7-2).
 *
 * Counterparts of file_handling::PointcloudFileHandler (XyzFileReader, PcdFileReader, BinvoxFileReader) and
 * geometry_generation::createBoxOfPoints. No CUDA, thrust, Eigen or PCL; differences are listed in
 * Agent/knowledge_base/architecture/vulkan_gpu_voxels_files.md.
 */
#ifndef GPU_VOXELS_VULKAN_POINT_CLOUD_FILE_H_INCLUDED
#define GPU_VOXELS_VULKAN_POINT_CLOUD_FILE_H_INCLUDED

#include <cstdint>
#include <filesystem>
#include <vector>

#include <gpu_voxels/vulkan/vulkan_common.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	/// A file that cannot be opened, has an unknown extension or is malformed (CUDA: logs and returns false). Derived from
	/// VulkanError; GpuVoxels::insert_point_cloud_from_file turns exactly this one into `false`.
	class PointCloudFileError : public VulkanError
	{
	public:
		using VulkanError::VulkanError;
	};

	/// Plain text, one point per line: the first three numbers of the line. Empty lines and lines that do not start with three
	/// numbers (comments, a header like "x y z") are skipped, further columns (normals, colours) are ignored (CUDA reads every
	/// triple of a line as a point, so a line with six numbers gives two points).
	[[nodiscard]] std::vector<Point3f> read_xyz_file(const std::filesystem::path& file);

	/// PCL point cloud data v0.7 with DATA ascii or DATA binary (not binary_compressed). Needs the fields x, y, z of type F (size 4
	/// or 8, count 1); other fields are skipped. All POINTS points are returned, also NaN ones of organised clouds (CUDA through PCL).
	[[nodiscard]] std::vector<Point3f> read_pcd_file(const std::filesystem::path& file);

	/// Binvox (any version of the header with dim / translate / scale / data, run length coded): one point per filled voxel,
	/// `scale / width * index + translate` per axis, x outermost, then z, then y (as CUDA). A file whose data ends before the
	/// volume is filled or has a run beyond it is an error (CUDA: the first is silently accepted).
	[[nodiscard]] std::vector<Point3f> read_binvox_file(const std::filesystem::path& file);

	/// Reader by extension (.xyz, .pcd, .binvox, case insensitive). \throws PointCloudFileError for any other extension
	[[nodiscard]] std::vector<Point3f> read_point_cloud_file(const std::filesystem::path& file);

	/// Moves the cloud so that its minimum on every axis is 0 (CUDA shiftPointCloudToZero). Points that are not finite are
	/// left alone and do not take part in the minimum; an empty cloud stays empty (CUDA: undefined).
	void shift_point_cloud_to_zero(std::vector<Point3f>& points);

	struct PointCloudLoadOptions
	{
		bool shift_to_zero = false;
		Point3f offset{ 0.0f, 0.0f, 0.0f };
		float scaling = 1.0f;
	};

	/// CUDA PointcloudFileHandler::loadPointCloud: reads `model_path / file` (an absolute `file` replaces `model_path`), shifts to
	/// zero if asked, then `p = scaling * p + offset` (so the offset is added after the shift and the scaling).
	[[nodiscard]] std::vector<Point3f> load_point_cloud(const std::filesystem::path& file, const std::filesystem::path& model_path = {},
		const PointCloudLoadOptions& options = {});

	/// Voxel coordinates of a box (CUDA createBoxOfPoints with a voxel side length).
	struct BoxVoxels
	{
		std::vector<Coord3ui> coordinates; ///< x outermost, z innermost; only voxels below `limit` on every axis
		uint64_t total = 0;                ///< voxels of the box before clipping to the limit
	};

	/// Most coordinates one call generates (after clipping to the limit).
	inline constexpr uint64_t kMaxBoxVoxels = uint64_t(1) << 26;

	/// The voxels from floor(max(min, 0) / side) to ceil(max / side) INCLUSIVE on every axis (the upper corner voxel is included
	/// even when max is a multiple of the side), stepping round(spacing / side) voxels (at least 1; CUDA hangs for 0) in float
	/// arithmetic like CUDA. Voxels at or beyond `limit` are not generated (and counted in `total` only).
	/// \throws VulkanError for a side length or spacing that is not positive and finite, a corner that is not finite or has a
	///         voxel coordinate beyond +-2^31, a box that is empty (voxel minimum above maximum; CUDA throws std::exception), or
	///         more than kMaxBoxVoxels voxels inside the limit
	[[nodiscard]] BoxVoxels box_voxel_coordinates(Point3f min, Point3f max, float spacing, float voxel_side_length, Dim3ui limit);
}

#endif
