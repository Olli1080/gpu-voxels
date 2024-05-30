#include "TemplateVoxelMap.cuhpp"

#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/DistanceVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>

#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels
{
	namespace voxelmap
	{
		template class TemplateVoxelMap<BitVoxel<BIT_VECTOR_LENGTH>>;
		template class TemplateVoxelMap<DistanceVoxel>;
		template class TemplateVoxelMap<ProbabilisticVoxel>;
	}
}