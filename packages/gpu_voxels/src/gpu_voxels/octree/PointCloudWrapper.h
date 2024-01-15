#pragma once

#include <cstdint>

#include "DataTypes.h"

namespace gpu_voxels
{
    namespace NTree
    {
        class Voxel;
        struct Sensor;

        void kernel_transformKinectPoints_simple_start(uint32_t num_blocks, uint32_t num_threads,
            gpu_voxels::Vector3f* d_point_cloud, const gpu_voxels::NTree::voxel_count num_points,
            gpu_voxels::OctreeVoxelID* d_tmp_voxel_id, gpu_voxels::NTree::Sensor* d_sensor, const uint32_t resolution);

        void kernel_transformKinectPoints_start(
            gpu_voxels::Vector3f* d_point_cloud, const gpu_voxels::NTree::voxel_count num_points,
            gpu_voxels::NTree::Voxel* d_tmp_voxel, gpu_voxels::NTree::Sensor* d_sensor, gpu_voxels::Vector3f voxel_dimension);

        void kernel_countVoxel_start(Voxel* d_tmp_voxel, voxel_count num_points, OctreeVoxelID* count_voxel);

        void kernel_combineEqualVoxel_start(Voxel* d_tmp_voxel, OctreeVoxelID num_voxel, OctreeVoxelID* count_voxel, Voxel* voxel, gpu_voxels::NTree::Sensor* d_sensor);

        template<bool T>
        void kernel_voxelize_start(uint32_t num_blocks, OctreeVoxelID* d_tmp_voxel_id, voxel_count num_points, voxel_count* count_voxel, Voxel* maybe_output);

        template void kernel_voxelize_start<false>(uint32_t num_blocks, OctreeVoxelID* d_tmp_voxel_id, voxel_count num_points, voxel_count* count_voxel, Voxel* maybe_output);
        template void kernel_voxelize_start<true>(uint32_t num_blocks, OctreeVoxelID* d_tmp_voxel_id, voxel_count num_points, voxel_count* count_voxel, Voxel* maybe_output);


        void kernel_voxelize_finalStep_start(uint32_t num_blocks, uint32_t num_threads,
            OctreeVoxelID* d_tmp_voxel_id, voxel_count num_points, voxel_count num_voxel, Voxel* d_voxel, gpu_voxels::NTree::Sensor* d_sensor);

    }
}