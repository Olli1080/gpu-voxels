// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Plain value types of the Vulkan backend API (no Eigen dependency yet).
 */
#ifndef GPU_VOXELS_VULKAN_TYPES_H_INCLUDED
#define GPU_VOXELS_VULKAN_TYPES_H_INCLUDED

#include <array>
#include <cstdint>

namespace gpu_voxels::vulkan
{
	/// Mirrors gpu_voxels::BitVoxelMeaning (helpers/common_defines.h); the numeric value is the bit index in the voxel.
	enum BitVoxelMeaning : uint8_t
	{
		eBVM_FREE = 0,
		eBVM_OCCUPIED = 1,
		eBVM_COLLISION = 2,
		eBVM_UNKNOWN = 3,
		eBVM_SWEPT_VOLUME_START = 4,
		eBVM_SWEPT_VOLUME_END = 254,
		eBVM_UNCERTAIN_OCC_PROB = 129, ///< ProbVoxelMap: swept volume meaning m adds m - 129 to the occupancy
		eBVM_UNDEFINED = 255
	};

	/// Mirrors gpu_voxels::MapType (helpers/common_defines.h), same order.
	enum MapType : uint8_t
	{
		MT_BITVECTOR_VOXELMAP,
		MT_BITVECTOR_VOXELLIST,
		MT_BITVECTOR_OCTREE,
		MT_BITVECTOR_MORTON_VOXELLIST, ///< not built (as in CUDA: "not implemented")
		MT_PROBAB_VOXELMAP,
		MT_PROBAB_VOXELLIST,           ///< not built (as in CUDA: "not implemented")
		MT_PROBAB_OCTREE,
		MT_PROBAB_MORTON_VOXELLIST,    ///< not built (as in CUDA: "not implemented")
		MT_COUNTING_VOXELLIST,
		MT_DISTANCE_VOXELMAP
	};

	struct Point3f { float x, y, z; };
	struct Coord3ui { uint32_t x, y, z; };
	struct Dim3ui
	{
		uint32_t x, y, z;
		bool operator==(const Dim3ui&) const = default;
		[[nodiscard]] uint64_t voxel_count() const { return static_cast<uint64_t>(x) * y * z; }
	};

	/// 256-bit vector of a BitVoxel as 8 words; bit i = meaning i (word i / 32, bit i % 32).
	using BitVector256 = std::array<uint32_t, 8>;

	/// 4x4 matrix, row-major: m[row * 4 + column]. A point is transformed like Eigen's
	/// `M.block<3,3>(0,0) * p + M.block<3,1>(0,3)` (the last row is ignored).
	struct Matrix4f
	{
		std::array<float, 16> m{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

		[[nodiscard]] static Matrix4f identity() { return {}; }
		float& operator()(int row, int column) { return m[static_cast<std::size_t>(row) * 4 + column]; }
		float operator()(int row, int column) const { return m[static_cast<std::size_t>(row) * 4 + column]; }
		bool operator==(const Matrix4f&) const = default;

		[[nodiscard]] Matrix4f operator*(const Matrix4f& o) const
		{
			Matrix4f r;
			for (int i = 0; i < 4; ++i)
				for (int j = 0; j < 4; ++j)
				{
					float sum = 0.0f;
					for (int k = 0; k < 4; ++k)
						sum += (*this)(i, k) * o(k, j);
					r(i, j) = sum;
				}
			return r;
		}
	};
}

#endif
