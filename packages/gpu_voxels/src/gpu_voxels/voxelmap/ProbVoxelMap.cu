#include "ProbVoxelMap.h"

#include <gpu_voxels/voxelmap/TemplateVoxelMap.cuhpp>
#include <gpu_voxels/voxel/BitVoxel.h>

namespace gpu_voxels {
    namespace voxelmap {

        ProbVoxelMap::ProbVoxelMap(Vector3ui dim, float voxel_side_length, MapType map_type)
            : Base(dim, voxel_side_length, map_type)
        {}

        ProbVoxelMap::ProbVoxelMap(Voxel* dev_data, Vector3ui dim, float voxel_side_length, MapType map_type)
            : Base(dev_data, dim, voxel_side_length, map_type)
        {}

        ProbVoxelMap::~ProbVoxelMap() = default;


        bool ProbVoxelMap::insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud* robot_links,
            const std::vector<BitVoxelMeaning>& voxel_meanings,
            const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks,
            BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
        {
            LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
            return true;
        }

        void ProbVoxelMap::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
        {
            if (voxel_meaning != eBVM_OCCUPIED)
                LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
            else
                this->clearMap();
        }

        //Collsion Interface Implementations

        size_t ProbVoxelMap::collideWith(const BitVectorVoxelMap* map, float coll_threshold, const Vector3i& offset)
        {
            DefaultCollider collider(coll_threshold);
            return collisionCheckWithCounterRelativeTransform((TemplateVoxelMap*)map, collider, offset); //does the locking

        }

        size_t ProbVoxelMap::collideWith(const ProbVoxelMap* map, float coll_threshold, const Vector3i& offset)
        {
            DefaultCollider collider(coll_threshold);
            return collisionCheckWithCounterRelativeTransform((TemplateVoxelMap*)map, collider, offset); //does the locking
        }
    }
}