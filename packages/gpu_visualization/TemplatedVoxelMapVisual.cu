#include "TemplatedVoxelMapVisual.hpp"

namespace gpu_voxels
{
	namespace voxelmap
	{
		ProbVoxelMapVisual::ProbVoxelMapVisual(Voxel* dev_data, const Vector3ui& dim, float voxel_side_length, MapType map_type)
			: Base(dev_data, dim, voxel_side_length, map_type)
		{}

		DistanceVoxelMapVisual::DistanceVoxelMapVisual(Voxel* dev_data, const Vector3ui& dim, float voxel_side_length, MapType map_type)
			: Base(dev_data, dim, voxel_side_length, map_type)
		{}

		MapType GpuVoxelsMapVisual::getMapType() const
		{
			return m_map_type;
		}
	}
}