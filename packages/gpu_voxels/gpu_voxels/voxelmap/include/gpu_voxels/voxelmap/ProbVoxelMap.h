// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-07-08
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_H_INCLUDED

#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>
#include <gpu_voxels/helpers/CollisionInterfaces.h>

namespace gpu_voxels
{
    namespace voxelmap
	{
        class ProbVoxelMap : public TemplateVoxelMap<ProbabilisticVoxel>,
            public CollidableWithBitVectorVoxelMap, public CollidableWithProbVoxelMap
        {
        public:

            typedef ProbabilisticVoxel Voxel;
            typedef TemplateVoxelMap<Voxel> Base;

            ProbVoxelMap(Vector3ui dim, float voxel_side_length, MapType map_type);
            ProbVoxelMap(Voxel* dev_data, Vector3ui dim, float voxel_side_length, MapType map_type);
            ~ProbVoxelMap() override;

            template<std::size_t length>
            void insertSensorData(const PointCloud& global_points, const Vector3f& sensor_pose, bool enable_raycasting, bool cut_real_robot,
                                  BitVoxelMeaning robot_voxel_meaning, BitVoxel<length>* robot_map = nullptr);

            bool insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud* robot_links,
                const std::vector<BitVoxelMeaning>& voxel_meanings = {},
                const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks = {},
                BitVector<BIT_VECTOR_LENGTH>* colliding_meanings = nullptr) override;

            void clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning) override;

            MapType getTemplateType() const override { return MT_PROBAB_VOXELMAP; }

            // Collision Interface Methods

            size_t collideWith(const voxelmap::BitVectorVoxelMap* map, float coll_threshold = 1.f, const Vector3i& offset = Vector3i::Zero()) override;
            size_t collideWith(const voxelmap::ProbVoxelMap* map, float coll_threshold = 1.f, const Vector3i& offset = Vector3i::Zero()) override;
        };

    } // end of namespace
} // end of namespace
#endif