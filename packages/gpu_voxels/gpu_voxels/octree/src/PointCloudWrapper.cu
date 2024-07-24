#include "PointCloudWrapper.h"

#include "kernels/kernel_PointCloud.h"

#include <gpu_voxels/octree/CommonValues.h>

namespace gpu_voxels
{
    namespace NTree
    {
		void kernel_transformKinectPoints_simple_start(uint32_t num_blocks, uint32_t num_threads, 
		    gpu_voxels::Vector3f* d_point_cloud, const voxel_count num_points, 
		    OctreeVoxelID* d_tmp_voxel_id, Sensor* d_sensor, const uint32_t resolution)
		{
			kernel_transformKinectPoints_simple<<<num_blocks, num_threads>>>(d_point_cloud, num_points,
		                                                                   d_tmp_voxel_id,
		                                                                   d_sensor,
		                                                                   resolution);
		}

		void kernel_transformKinectPoints_start(
		    gpu_voxels::Vector3f* d_point_cloud, const voxel_count num_points, 
		    Voxel* d_tmp_voxel, Sensor* d_sensor, gpu_voxels::Vector3f voxel_dimension)
		{
			kernel_transformKinectPoints<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_point_cloud, num_points, d_tmp_voxel, d_sensor, voxel_dimension);
		}
		    
		void kernel_countVoxel_start(Voxel* d_tmp_voxel, voxel_count num_points, OctreeVoxelID* count_voxel)
		{
			kernel_countVoxel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_tmp_voxel, num_points, count_voxel);
		}

		void kernel_combineEqualVoxel_start(Voxel* d_tmp_voxel, OctreeVoxelID num_voxel, OctreeVoxelID* count_voxel, Voxel* voxel, gpu_voxels::NTree::Sensor* d_sensor)
		{
			kernel_combineEqualVoxel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_tmp_voxel, num_voxel, count_voxel, voxel, d_sensor);
		}

		void kernel_voxelize_finalStep_start(uint32_t num_blocks, uint32_t num_threads, 
		    OctreeVoxelID* d_tmp_voxel_id, voxel_count num_points, voxel_count num_voxel, Voxel* d_voxel, gpu_voxels::NTree::Sensor* d_sensor)
		{
			kernel_voxelize_finalStep<<<num_blocks, num_threads>>>(d_tmp_voxel_id,
			  num_points,
			  num_voxel,
			  d_voxel,
			  d_sensor);
		}

		template<bool T>
		void kernel_voxelize_start(uint32_t num_blocks, OctreeVoxelID* d_tmp_voxel_id, voxel_count num_points, voxel_count* count_voxel, Voxel* maybe_output)
		{
		    kernel_voxelize<T> <<<num_blocks, 32>>>(d_tmp_voxel_id, num_points, count_voxel, nullptr);
		}
    }
}
