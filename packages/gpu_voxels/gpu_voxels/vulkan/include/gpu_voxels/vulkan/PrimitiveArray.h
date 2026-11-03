// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Host container of primitives (spheres, cuboids) for a viewer (slice S7-5 of the migration plan).
 *
 * Counterpart of gpu_voxels::primitive_array::PrimitiveArray (and the primitives members of GpuVoxels). The CUDA class copies
 * the positions to device memory for the OpenGL viewer; nothing on the GPU reads them, so this class only keeps them on the
 * host, in the same representation the CUDA class stores: per primitive the centre in voxel units (the centre of the voxel
 * that contains the point) and half of the size in voxel units.
 */
#ifndef GPU_VOXELS_VULKAN_PRIMITIVE_ARRAY_H_INCLUDED
#define GPU_VOXELS_VULKAN_PRIMITIVE_ARRAY_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	/// Mirrors gpu_voxels::primitive_array::PrimitiveType (ePRIM_INITIAL_VALUE is an internal marker of the CUDA class and not offered).
	enum PrimitiveType : uint8_t
	{
		ePRIM_SPHERE = 0,
		ePRIM_CUBOID = 1
	};

	/// Position in metres and size (diameter of a sphere, edge of a cuboid) in metres (CUDA Vector4f).
	struct Point4f { float x, y, z, w; };
	/// Position in voxels and size in voxels (CUDA Vector3i / Vector4i).
	struct Coord3i { int32_t x, y, z; };
	struct Coord4i { int32_t x, y, z, w; };

	/// One stored primitive: centre in voxel units and half of the size in voxel units (CUDA stores exactly these four floats).
	struct PrimitiveEntry
	{
		float x, y, z, radius;
		bool operator==(const PrimitiveEntry&) const = default;
	};

	class PrimitiveArray
	{
	public:
		/// \param dimensions kept for the viewer, not used here (as in CUDA)
		/// \throws VulkanError for a type other than the two above or a voxel side length that is not positive and finite
		PrimitiveArray(Dim3ui dimensions, float voxel_side_length, PrimitiveType type);

		/// Replaces all primitives. Metric positions become the centre of their voxel: floor(p / side) + 0.5, the size
		/// (w / side) / 2, computed in float like CUDA; voxel positions p + 0.5 and w / 2. Positions are not checked against the map.
		/// \throws VulkanError (nothing changes) for a primitive whose position or size is not finite in voxel units, or whose size is negative
		void set_points(std::span<const Point4f> metric_points);
		void set_points(std::span<const Coord4i> voxel_points);
		/// All primitives with the same size (metres / voxels).
		void set_points(std::span<const Point3f> metric_points, float diameter);
		void set_points(std::span<const Coord3i> voxel_points, uint32_t diameter);

		[[nodiscard]] std::span<const PrimitiveEntry> primitives() const { return primitives_; }
		[[nodiscard]] std::size_t size() const { return primitives_.size(); }
		[[nodiscard]] PrimitiveType type() const { return type_; }
		[[nodiscard]] Dim3ui dimensions() const { return dimensions_; }
		[[nodiscard]] float voxel_side_length() const { return voxel_side_length_; }
		/// CUDA getMemoryUsage: the device memory of the CUDA class, 16 bytes per primitive.
		[[nodiscard]] std::size_t memory_usage_bytes() const { return primitives_.size() * sizeof(PrimitiveEntry); }

	private:
		void store(std::vector<PrimitiveEntry>&& primitives);

		Dim3ui dimensions_;
		float voxel_side_length_;
		PrimitiveType type_;
		std::vector<PrimitiveEntry> primitives_;
	};

	using PrimitiveArraySharedPtr = std::shared_ptr<PrimitiveArray>;
}

#endif
