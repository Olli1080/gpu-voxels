#include "BitVoxelList.hpp"

#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels
{
	namespace voxellist
	{
		template struct BitvectorCollision<BIT_VECTOR_LENGTH>;
		template struct BitvectorCollisionWithBitshift<BIT_VECTOR_LENGTH>;
		template struct BitvectorOr<BIT_VECTOR_LENGTH>;
		template struct ShiftBitvector<BIT_VECTOR_LENGTH>;

		template class BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>;
		//template class BitVoxelList<BIT_VECTOR_LENGTH, OctreeVoxelID>;

		template size_t BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>::collideWithTypeMask(const TemplateVoxelMap<BitVoxel<BIT_VECTOR_LENGTH>>* map, const BitVoxel<BIT_VECTOR_LENGTH>& types_to_check, float coll_threshold, const Vector3i& offset);
		template size_t BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>::collideWithTypeMask(const TemplateVoxelMap<CountingVoxel>* map, const BitVoxel<BIT_VECTOR_LENGTH>& types_to_check, float coll_threshold, const Vector3i& offset);
		template size_t BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>::collideWithTypeMask(const TemplateVoxelMap<ProbabilisticVoxel>* map, const BitVoxel<BIT_VECTOR_LENGTH>& types_to_check, float coll_threshold, const Vector3i& offset);
		//template size_t BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>::collideWithTypeMask(const TemplateVoxelMap<DistanceVoxel>* map, const BitVoxel<BIT_VECTOR_LENGTH>& types_to_check, float coll_threshold, const Vector3i& offset);
		/*
		template size_t BitVoxelList<BIT_VECTOR_LENGTH, OctreeVoxelID>::collideWithTypeMask(const TemplateVoxelMap<BitVoxel<BIT_VECTOR_LENGTH>>* map, const BitVoxel<BIT_VECTOR_LENGTH>& types_to_check, float coll_threshold, const Vector3i& offset);
		template size_t BitVoxelList<BIT_VECTOR_LENGTH, OctreeVoxelID>::collideWithTypeMask(const TemplateVoxelMap<CountingVoxel>* map, const BitVoxel<BIT_VECTOR_LENGTH>& types_to_check, float coll_threshold, const Vector3i& offset);
		template size_t BitVoxelList<BIT_VECTOR_LENGTH, OctreeVoxelID>::collideWithTypeMask(const TemplateVoxelMap<ProbabilisticVoxel>* map, const BitVoxel<BIT_VECTOR_LENGTH>& types_to_check, float coll_threshold, const Vector3i& offset);
		template size_t BitVoxelList<BIT_VECTOR_LENGTH, OctreeVoxelID>::collideWithTypeMask(const TemplateVoxelMap<DistanceVoxel>* map, const BitVoxel<BIT_VECTOR_LENGTH>& types_to_check, float coll_threshold, const Vector3i& offset);
		*/
	}
}