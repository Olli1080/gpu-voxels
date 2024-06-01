// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
 //----------------------------------------------------------------------
 //#define LOCAL_DEBUG
#undef LOCAL_DEBUG

#include "VoxelMapOperations.hpp"

#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include "gpu_voxels/helpers/BitVector.h"
#include <gpu_voxels/voxel/BitVoxel.cuhpp>
#include <cstdio>

namespace gpu_voxels {
	namespace voxelmap {

		//DistanceVoxel specialization
		template<>
		__global__
		void kernelInsertGlobalPointCloud(DistanceVoxel* voxelmap, Vector3ui map_dim, float voxel_side_length,
			const Vector3f* points, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
			bool* points_outside_map)
		{

			//debug
		  //  if (blockIdx.x + threadIdx.x == 0) {
		  //    printf("DEBUG: DistanceVoxelMap::insertPointCloud was called instead of the TemplateVoxelMap one\n");
		  //    for (uint32_t i = 0; i < sizePoints; i += 1) {
		  //      printf("DEBUG: point %u: %f %f %f\n", i, points[i].x, points[i].y, points[i].z);
		  //    }
		  //  }

			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
			{
				const Vector3ui uint_coords = mapToVoxels(voxel_side_length, points[i]);
				//check if point is in the range of the voxel map
				if (uint_coords.x() < map_dim.x() && uint_coords.y() < map_dim.y()
					&& uint_coords.z() < map_dim.z())
				{
					DistanceVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(map_dim,
						uint_coords.x(), uint_coords.y(), uint_coords.z())];
					voxel->insert(uint_coords, voxel_meaning);
				}
				else
				{
					if (points_outside_map) *points_outside_map = true;
					//      printf("DistanceVoxel kernelInsertGlobalPointCloud: Point (%u,%u,%u) is not in the range of the voxel map \n",
					//             points[i].x, points[i].y, points[i].z);
				}
			}
		}

		// DistanceVoxel specialization
		template<>
		__global__
		void kernelInsertCoordinateTuples(DistanceVoxel* voxelmap, Vector3ui map_dim, float voxel_side_length,
			const Vector3ui* coordinates, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
			bool* points_outside_map)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
			{
				const Vector3ui uint_coords = coordinates[i];
				//check if point is in the range of the voxel map
				if (uint_coords.x() < map_dim.x() && uint_coords.y() < map_dim.y()
					&& uint_coords.z() < map_dim.z())
				{
					DistanceVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(map_dim, uint_coords)];
					voxel->insert(uint_coords, voxel_meaning);
				}
				else
				{
					if (points_outside_map) *points_outside_map = true;
					//       printf("Point (%u,%u,%u) is not in the range of the voxel map \n", points[i].x, points[i].y,
					//              points[i].z);getVoxelIndexUnsigned
				}
			}
		}


		//DistanceVoxel specialization
		template<>
		__global__
		void kernelInsertMetaPointCloud(DistanceVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			BitVoxelMeaning voxel_meaning, Vector3ui dimensions, float voxel_side_length,
			bool* points_outside_map)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
				i += blockDim.x * gridDim.x)
			{
				const Vector3ui uint_coordinates = mapToVoxels(voxel_side_length,
					meta_point_cloud->clouds_base_addresses[0][i]);

				//        printf("Point @(%f,%f,%f)\n",
				//               meta_point_cloud->clouds_base_addresses[0][i].x,
				//               meta_point_cloud->clouds_base_addresses[0][i].y,
				//               meta_point_cloud->clouds_base_addresses[0][i].z);

					//check if point is in the range of the voxel map
				if (uint_coordinates.x() < dimensions.x() && uint_coordinates.y() < dimensions.y()
					&& uint_coordinates.z() < dimensions.z())
				{
					DistanceVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions,
						uint_coordinates.x(), uint_coordinates.y(), uint_coordinates.z())];
					voxel->insert(uint_coordinates, voxel_meaning);

					//        printf("Inserted Point @(%u,%u,%u) into the voxel map \n",
					//               integer_coordinates.x,
					//               integer_coordinates.y,
					//               integer_coordinates.z);

				}
				else
				{
					if (points_outside_map) *points_outside_map = true;
					//      printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
					//             meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
					//             meta_point_cloud->clouds_base_addresses[0][i].z);
				}
			}
		}


		//BitVectorVoxel specialization
		// This kernel may not be called with more threads than point per subcloud, as otherwise we will miss selfcollisions!
		template<>
		__global__
		void kernelInsertMetaPointCloudSelfCollCheck(BitVectorVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			const BitVoxelMeaning* voxel_meanings, Vector3ui dimensions, unsigned int sub_cloud,
			float voxel_side_length, const BitVector<BIT_VECTOR_LENGTH>* coll_masks,
			bool* points_outside_map, BitVector<BIT_VECTOR_LENGTH>* colliding_subclouds)
		{
			BitVector<BIT_VECTOR_LENGTH> masked;

			const uint32_t sub_cloud_upper_bound = meta_point_cloud->cloud_sizes[sub_cloud];

			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sub_cloud_upper_bound;
				i += blockDim.x * gridDim.x)
			{
				Vector3ui uint_coords = mapToVoxels(voxel_side_length, meta_point_cloud->clouds_base_addresses[sub_cloud][i]);

				//        printf("Point @(%f,%f,%f)\n",
				//               meta_point_cloud->clouds_base_addresses[0][i].x,
				//               meta_point_cloud->clouds_base_addresses[0][i].y,
				//               meta_point_cloud->clouds_base_addresses[0][i].z);

					//check if point is in the range of the voxel map
				if (uint_coords.x() < dimensions.x() && uint_coords.y() < dimensions.y() && uint_coords.z() < dimensions.z())
				{
					BitVectorVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, uint_coords)];
					masked.clear();
					masked = voxel->bitVector() & coll_masks[sub_cloud];
					if (!masked.noneButEmpty())
					{
						*colliding_subclouds |= masked; // copy the meanings of the colliding voxel, except the masked ones
						colliding_subclouds->setBit(voxel_meanings[sub_cloud]); // also set collisions for own meaning
						voxel->insert(eBVM_COLLISION); // Mark voxel as colliding
					}

					voxel->insert(voxel_meanings[sub_cloud]); // insert subclouds point


					//        printf("Inserted Point @(%u,%u,%u) with meaning %u into the voxel map \n",
					//               integer_coordinates.x,
					//               integer_coordinates.y,
					//               integer_coordinates.z,
					//               voxel_meanings[voxel_meaning_index]);

				}
				else
				{
					if (points_outside_map) *points_outside_map = true;
					//       printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
					//              meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
					//              meta_point_cloud->clouds_base_addresses[0][i].z);
				}

			} // grid stride loop
		  //    unsigned int foo = atomicInc(global_sub_cloud_control, *global_sub_cloud_control);
		  //    printf("This thread inserted point %d, which was last of subcloud. Incrementing global control value to %d ...\n", i, (foo+1));
		}


		//DistanceVoxel specialization
		template<>
		__global__
		void kernelInsertMetaPointCloud(DistanceVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			BitVoxelMeaning* voxel_meanings, Vector3ui map_dim,
			float voxel_side_length,
			bool* points_outside_map)
		{
			uint16_t sub_cloud = 0;
			uint32_t sub_cloud_upper_bound = meta_point_cloud->cloud_sizes[sub_cloud];

			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
				i += blockDim.x * gridDim.x)
			{
				// find out, to which sub_cloud our point belongs
				while (i >= sub_cloud_upper_bound)
				{
					sub_cloud++;
					sub_cloud_upper_bound += meta_point_cloud->cloud_sizes[sub_cloud];
				}


				const Vector3ui uint_coordinates = mapToVoxels(voxel_side_length,
					meta_point_cloud->clouds_base_addresses[0][i]);

				//        printf("Point @(%f,%f,%f)\n",
				//               meta_point_cloud->clouds_base_addresses[0][i].x,
				//               meta_point_cloud->clouds_base_addresses[0][i].y,
				//               meta_point_cloud->clouds_base_addresses[0][i].z);

					//check if point is in the range of the voxel map
				if (uint_coordinates.x() < map_dim.x() && uint_coordinates.y() < map_dim.y()
					&& uint_coordinates.z() < map_dim.z())
				{
					DistanceVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(map_dim,
						uint_coordinates.x(), uint_coordinates.y(), uint_coordinates.z())];
					voxel->insert(uint_coordinates, voxel_meanings[sub_cloud]);

					//        printf("Inserted Point @(%u,%u,%u) with meaning %u into the voxel map \n",
					//               integer_coordinates.x,
					//               integer_coordinates.y,
					//               integer_coordinates.z,
					//               voxel_meanings[voxel_meaning_index]);

				}
				else
				{
					if (points_outside_map) *points_outside_map = true;
					/* printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
						   meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
						   meta_point_cloud->clouds_base_addresses[0][i].z); */
				}
			}
		}


		/**
		 * cjuelg: jump flood distances, obstacle vectors
		 *
		 *
		 * algorithm:
		 *  calcNearestObstaclesJFA(VoxelMap, dim3, uint step_num (log(maxdim)..1))
		 *       set map[x,y,z]= min(pos+{-1,0,1}*{x,y,z})
		 */
		__global__
		void kernelJumpFlood3D(const DistanceVoxel* __restrict__ const voxels_input, DistanceVoxel* __restrict__ const voxels_output, const Vector3ui dims, const int32_t step_width)
		{
			const uint32_t numVoxels = dims.x() * dims.y() * dims.z();

			//get linear address i
			//repeat if grid.x*block.x < numVoxels
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numVoxels; i += blockDim.x * gridDim.x)
			{
				//map to x,y,z
				Vector3i pos;
				pos.x() = i % dims.x();
				pos.y() = i / dims.x() % dims.y();
				pos.z() = i / (dims.x() * dims.y()) % dims.z();

				//check if point is in the range of the voxel map
				if (pos.x() < dims.x() && pos.y() < dims.y() && pos.z() < dims.z()) // should always be true
				{
					DistanceVoxel min_voxel = voxels_input[i];

					if (min_voxel.squaredObstacleDistance(pos) == PBA_OBSTACLE_DISTANCE) {
						voxels_output[i] = min_voxel; // copy to buffer
						continue; //no other obstacle can be closer
					}

					// load 26 "step-neighbors"; for each: if distance is smaller, save obstacle and distance; (reduction operation)
					for (int x_step = -step_width; x_step <= step_width; x_step += step_width) {

						const int x_check = pos.x() + x_step;
						if (x_check >= 0 && x_check < dims.x()) { //don't leave map limits

							for (int y_step = -step_width; y_step <= step_width; y_step += step_width) {

								const int y_check = pos.y() + y_step;
								if (y_check >= 0 && y_check < dims.y()) { //don't leave map limits

									for (int z_step = -step_width; z_step <= step_width; z_step += step_width) {

										const int z_check = pos.z() + z_step;
										if (z_check >= 0 && z_check < dims.z()) { //don't leave map limits

											if (x_step != 0 || y_step != 0 || z_step != 0) { //don't compare center_voxel to self
												updateMinVoxel(voxels_input[getVoxelIndexSigned(dims, x_check, y_check, z_check)], min_voxel, pos);
											}
										}
									}
								}
							}
						}
					}

					voxels_output[i] = min_voxel; //always update output array, even if min_voxel = voxels_input[i]
				}
				else
				{
					printf("(%i,%i,%i) is not in the range of the voxel map; SHOULD BE IMPOSSIBLE \n", pos.x(), pos.y(), pos.z());
				}
			}
		}


		/**
		 * cjuelg: brute force exact obstacle distances
		 *
		 * optimization1: check against known obstacle list instead of all other voxels
		 * optimization2: use shared memory to prefetch chunks of the obstacle list in parallel
		 * optimization3: resolve bank conflicts by using threadIdx as offset?
		 */
		__global__
		void kernelExactDistances3D(DistanceVoxel* voxels, Vector3ui dims, float voxel_side_length,
			Vector3f* obstacles, std::size_t num_obstacles)
		{
			extern __shared__ int dynamic_shared_mem[];
			auto* obstacle_cache = (Vector3i*)dynamic_shared_mem; //size: cMAX_THREADS_PER_BLOCK * sizeof(DistanceVoxel)

			const uint32_t num_voxels = dims.x() * dims.y() * dims.z();

			//get linear address i

			if (gridDim.x * gridDim.y * blockDim.x < num_voxels)
				printf("exactDifferences3D: Alert: grids and blocks don't span num_voxels!");


			uint32_t voxel_idx = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
			if (voxel_idx >= num_voxels) return;
			DistanceVoxel* pos_voxel = &voxels[voxel_idx];

			Vector3i pos;
			pos.x() = voxel_idx % dims.x();
			pos.y() = voxel_idx / dims.x() % dims.y();
			pos.z() = voxel_idx / (dims.x() * dims.y()) % dims.z();

			int32_t min_distance = pos_voxel->squaredObstacleDistance(pos);
			Vector3i min_obstacle;

			for (unsigned int obstacle_prefetch_offset = 0; obstacle_prefetch_offset < num_obstacles; obstacle_prefetch_offset += blockDim.x) {
				const unsigned int obstacle_prefetch_idx = obstacle_prefetch_offset + threadIdx.x;

				// prefetch
				if (obstacle_prefetch_idx < num_obstacles) {
					const Vector3i obstacle = mapToVoxelsSigned(voxel_side_length, obstacles[obstacle_prefetch_idx]);
					obstacle_cache[threadIdx.x] = obstacle;
				}
				__syncthreads();

				// update closest obstacle

				//check if point is in the range of the voxel map
				if (pos.x() < dims.x() && pos.y() < dims.y() && pos.z() < dims.z()) //always true?
				{
					if (min_distance != PBA_OBSTACLE_DISTANCE) { //else no other obstacle can be closer

						//check for every obstacle whether it is the closest one to pos
						for (unsigned int s_obstacle_idx = 0; s_obstacle_idx < cMAX_THREADS_PER_BLOCK && obstacle_prefetch_offset + s_obstacle_idx < num_obstacles; s_obstacle_idx++) {

							//optimise: resolve bank conflicts by using threadIdx as offset?
							//TODO: test optimisation, might even be slower (test using large obstacle count; with low obstacle count the kernel runs <2ms
							//          int cache_size = min(blockDim.x, (uint)num_obstacles - obstacle_prefetch_offset);
							//          const Vector3i obstacle_pos = obstacle_cache[(s_obstacle_idx + threadIdx.x ) % cache_size];
							const Vector3i obstacle_pos = obstacle_cache[s_obstacle_idx];

							//TODO: could perform sanity check, but: expensive, explodes number of memory accesses
				  //            const DistanceVoxel* other_voxel = &voxels[getVoxelIndex(dims, obstacle_pos.x, obstacle_pos.y, obstacle_pos.z)];
				  //            if (other_voxel->getDistance() != DISTANCE_OBSTACLE) {
				  //              printf("ERROR: exactDistances3D: (pos: %i,%i,%i) given obstacle coordinates do not contain obstacle: (%u,%u,%u), %d\n", pos.x, pos.y, pos.z, obstacle_pos.x, obstacle_pos.y, obstacle_pos.z, other_voxel->getDistance());
				  //            }
							//            if (other_voxel != center_voxel && other_voxel->getDistance() == DISTANCE_OBSTACLE) {

							if (obstacle_pos != pos) {
								int32_t other_distance;
								if (
									obstacle_pos.x() == PBA_UNINITIALISED_COORD
									|| obstacle_pos.y() == PBA_UNINITIALISED_COORD
									|| obstacle_pos.z() == PBA_UNINITIALISED_COORD
									|| pos.x() == PBA_UNINITIALISED_COORD
									|| pos.y() == PBA_UNINITIALISED_COORD
									|| pos.z() == PBA_UNINITIALISED_COORD
									)
								{
									other_distance = MAX_OBSTACLE_DISTANCE;

								}
								else {  // never use PBA_UNINIT in calculations
									const int dx = pos.x() - obstacle_pos.x(), dy = pos.y() - obstacle_pos.y(), dz = pos.z() - obstacle_pos.z();
									other_distance = dx * dx + dy * dy + dz * dz; //squared distance

									if (other_distance < min_distance) { //need to update minimum
										min_distance = other_distance;
										min_obstacle = obstacle_pos;
									}
								}
							}
						}
					}
				}
				else {
					printf("(%i,%i,%i) is not in the range of the voxel map; SHOULD BE IMPOSSIBLE \n", pos.x(), pos.y(), pos.z());
				}
				__syncthreads();
			}

			if (min_distance < pos_voxel->squaredObstacleDistance(pos)) //need to update pos_voxel
				pos_voxel->setObstacle(min_obstacle);
		}

		//
		//void kernelCalculateBoundingBox(Voxel* voxelmap, const uint32_t voxelmap_size, )

		//__global__
		//void kernelInsertKinematicLinkBitvector(Voxel* voxelmap, const uint32_t voxelmap_size,
		//                                        const Vector3ui dimensions, const float voxel_side_length,
		//                                        uint32_t link_nr, uint32_t* point_cloud_sizes,
		//                                        Vector3f** point_clouds, uint64_t bit_number)
		//{
		//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		//  const Vector3ui map_dim = (*dimensions);
		//
		//  if (i < point_cloud_sizes[link_nr])
		//  {
		//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, point_clouds[link_nr][i]);
		//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
		//        && (integer_coordinates.z < map_dim.z))
		//    {
		//
		//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
		//                                 integer_coordinates.z);
		//      voxel->setBitvector(voxel->getBitvector() | bit_number);
		//
		//    }
		//  }
		//}

		//__global__
		//void kernelInsertRobotKinematicLinkOverwritingSensorData(Voxel* voxelmap, const uint32_t voxelmap_size,
		//                                                         const Vector3ui dimensions,
		//                                                         const float voxel_side_length,
		//                                                         const MetaPointCloudStruct *robot_links,
		//                                                         uint32_t link_nr, const Voxel* environment_map)
		//{
		//  const Vector3ui map_dim = (*dimensions);
		//
		//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < robot_links->cloud_sizes[link_nr];
		//      i += gridDim.x * blockDim.x)
		//  {
		//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length,
		//                                                      robot_links->clouds_base_addresses[link_nr][i]);
		//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
		//        && (integer_coordinates.z < map_dim.z))
		//    {
		//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
		//                                 integer_coordinates.z);
		//      Voxel* env_voxel = getVoxelPtr(environment_map, dimensions, integer_coordinates.x,
		//                                     integer_coordinates.y, integer_coordinates.z);
		//      voxel->voxelmeaning = eBVM_OCCUPIED;
		//      voxel->occupancy = 255;
		//      env_voxel->occupancy = 0;
		//    }
		//  }
		//}

		///*! Insert a configuration for a kinematic link with self-collision check.
		// *  Always set self_ to false before calling this function because
		// *  it only indicates if there was a collision and not if there was none!
		// */
		//__global__
		//void kernelInsertRobotKinematicLinkWithSelfCollisionCheck(Voxel* voxelmap, const uint32_t voxelmap_size,
		//                                                          const Vector3ui dimensions,
		//                                                          const float voxel_side_length,
		//                                                          const MetaPointCloudStruct *robot_links,
		//                                                          uint32_t link_nr, bool* self_collision)
		//{
		//  const Vector3ui map_dim = (*dimensions);
		//
		//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < robot_links->cloud_sizes[link_nr];
		//      i += gridDim.x * blockDim.x)
		//  {
		//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length,
		//                                                      robot_links->clouds_base_addresses[link_nr][i]);
		//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
		//        && (integer_coordinates.z < map_dim.z))
		//    {
		//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
		//                                 integer_coordinates.z);
		//
		//      if (voxel->occupancy != 0)
		//      {
		//        (*self_collision) = true;
		//      }
		//      else
		//      {
		//        voxel->voxelmeaning = eBVM_OCCUPIED;
		//        voxel->occupancy = 255;
		//      }
		//    }
		//  }
		//}



		//__global__
		//void kernelCollideVoxelMapsBoundingBox(Voxel* voxelmap, const uint32_t voxelmap_size, const uint8_t threshold,
		//                                       Voxel* other_map, const uint8_t other_threshold, bool* results,
		//                                       uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
		//                                       uint32_t size_x, Vector3ui* dimensions)
		//{
		////  extern __shared__ bool cache[];//[cMAX_THREADS_PER_BLOCK];			//define Cache size in kernel call
		////  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		//
		//  //calculate i:
		//  uint32_t i = (offset_z + threadIdx.x) * dimensions->x * dimensions->y + //every Column on the y-axis is one block and
		//      (offset_y + blockIdx.x) * dimensions->x + offset_x; // the threads are going from z = 0 till dim_z
		//  uint32_t counter = 0;
		////  printf("thread idx: %i, blockIdx: %i, dimx: %i, dimy: %i", threadIdx.x, blockIdx.x, dimensions->x, dimensions->y);
		//
		////  uint32_t cache_index = threadIdx.x;
		////  int32_t _size_x = size_x;
		////  int32_t counter = 0;
		//
		////  cache[cache_index] = false;
		//  bool temp = false;
		//
		//  while (counter < size_x)
		//  {
		//    // todo / note: at the moment collision check is only used for DYNAMIC and SWEPT VOLUME meaning, static is used for debugging
		//    temp = temp
		//        || ((voxelmap[i].occupancy >= threshold) && (voxelmap[i].voxelmeaning != eBVM_OCCUPIED)
		//            && (other_map[i].occupancy >= other_threshold) && (other_map[i].voxelmeaning != eBVM_OCCUPIED));
		//
		//    counter += 1;
		//    i += 1;
		////      i += blockDim.x * gridDim.x;
		//  }
		//
		////  if(true)//(i == 30050600)
		////  {
		////	  printf("thread %i, collision %i \n", i, temp);
		////	  printf("--- occupation planning: %i, voxelmeaning planning: %i \n",
		////			  other_map[i].occupancy_planning, other_map[i].voxelmeaning_planning);
		////  }
		//
		//  results[blockIdx.x * blockDim.x + threadIdx.x] = temp; //
		//
		////  cache[cache_index] = temp;
		////  __syncthreads();
		////
		////  uint32_t j = blockDim.x / 2;
		////
		////  while (j!=0)
		////  {
		////    if (cache_index < j)
		////    {
		////      cache[cache_index] = cache[cache_index] || cache[cache_index + j];
		////    }
		////    __syncthreads();
		////    j /= 2;
		////  }
		////
		////  // copy results from this block to global memory
		////  if (cache_index == 0)
		////  {
		//////    // FOR MEASUREMENT TEMPORARILY EDITED:
		//////    results[blockIdx.x] = true;
		////    results[blockIdx.x] = cache[0];
		////  }
		//}



		//__global__
		//void kernelShrinkCopyVoxelMapBitvector(Voxel* destination_map, const uint32_t destination_map_size,
		//                                       Vector3ui* dest_map_dim, Voxel* source_map,
		//                                       const uint32_t source_map_size, Vector3ui* source_map_dim,
		//                                       uint8_t factor)
		//{
		//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		//  if (i < destination_map_size)
		//  {
		//    Voxel* v = destination_map + i;
		//    //getting indices for the Destination Voxel i
		//    Vector3ui dest_voxel_index = mapToVoxels(destination_map, dest_map_dim, v);
		//
		//    uint64_t bitvector = 0;
		//    //loop over every axis and get the value of the voxel
		//    Voxel* index_z = getVoxelPtr(source_map, source_map_dim, dest_voxel_index.x * factor,
		//                                 dest_voxel_index.y * factor, dest_voxel_index.z * factor);
		//    Voxel* index_y;
		//    Voxel* index_x;
		//    for (uint8_t z = 0; z < factor; ++z)
		//    {
		//      index_y = index_z; //resetting the index
		//      for (uint8_t y = 0; y < factor; ++y)
		//      {
		//        index_x = index_y;
		//        for (uint8_t x = 0; x < factor; ++x)
		//        {
		//          bitvector |= index_x->getBitvector();
		//          index_x += 1;
		//        }
		//        index_y += source_map_dim->x;
		//      }
		//      index_z += source_map_dim->y * source_map_dim->x;
		//    }
		//    v->setBitvector(bitvector);
		//  }
		//}


		////for different sized voxelmaps
		//__global__
		//void kernelShrinkCopyVoxelMap(Voxel* destination_map, const uint32_t destination_map_size,
		//                              Vector3ui* dest_map_dim, Voxel* source_map, const uint32_t source_map_size,
		//                              Vector3ui* source_map_dim, uint8_t factor)
		//{
		//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		//  if (i < destination_map_size)
		//  {
		//    Voxel* v = destination_map + i;
		//    //getting indices for the Destination Voxel i
		//    Vector3ui dest_voxel_index = mapToVoxels(destination_map, dest_map_dim, v);
		//
		//    uint8_t occupancy = 0;
		//    uint8_t voxelmeaning = 0;
		//    //loop over every axis and get the value of the voxel
		//    Voxel* index_z = getVoxelPtr(source_map, source_map_dim, dest_voxel_index.x * factor,
		//                                 dest_voxel_index.y * factor, dest_voxel_index.z * factor);
		//    Voxel* index_y;
		//    Voxel* index_x;
		//    for (uint8_t z = 0; z < factor; ++z)
		//    {
		//      index_y = index_z; //resetting the index
		//      for (uint8_t y = 0; y < factor; ++y)
		//      {
		//        index_x = index_y;
		//        for (uint8_t x = 0; x < factor; ++x)
		//        {
		//          if (index_x->occupancy > occupancy)
		//          {
		//            occupancy = index_x->occupancy;
		//            voxelmeaning = index_x->voxelmeaning;
		//          }
		//          index_x += 1;
		//        }
		//        index_y += source_map_dim->x;
		//      }
		//      index_z += source_map_dim->y * source_map_dim->x;
		//    }
		//    v->occupancy = occupancy;
		//    v->voxelmeaning = voxelmeaning;
		//  }
		//}

		thrust::device_vector<uint32_t> getOccupationVoxels(const thrust::device_vector<BitVoxel<BIT_VECTOR_LENGTH>>& dev_data)
		{
			thrust::device_vector<uint32_t> isVoxelOccupied(dev_data.size());
			thrust::transform(dev_data.begin(), dev_data.end(),
				isVoxelOccupied.begin(), OccupiedVoxels());

			return isVoxelOccupied;
		}

		uint32_t getOccupied(const thrust::device_vector<uint32_t>& vec)
		{
			return thrust::reduce(vec.begin(), vec.end(), 0);
		}

		__host__ __device__ uint32_t OccupiedVoxels::operator()(const BitVoxel<BIT_VECTOR_LENGTH>& val)
		{
			//TODO:: return 0 and 1 from isoccupied_numerical
			return (val.isOccupied(0)) ? 1 : 0;
		}

		uint32_t getOccupied_2(thrust::device_vector<uint32_t>& in_out)
		{
			thrust::device_vector<uint32_t> temp(in_out.size());
			thrust::inclusive_scan(in_out.begin(), in_out.end(), temp.begin());

			return temp.back();
		}


		CullHiddenVoxels::CullHiddenVoxels(const uint32_t* data, Vector3ui dims)
			: m_data(data), m_dim(std::move(dims))
		{}

		__host__ __device__ uint32_t CullHiddenVoxels::operator()(const uint32_t& idx)
		{
			if (m_data[idx] == 0)
				return 0;

			const Vector3ui xyz = indexToXYZ(idx, m_dim);
			//TODO:: maybe avoid divergence by grouping border voxels

			uint32_t res = 0;

			if (xyz.x() > 0)
				res += m_data[getVoxelIndexUnsigned(m_dim, xyz.x() - 1, xyz.y(), xyz.z())];
			//no else += 1, because we want to keep voxels at the boundary required that there are no negative indices

			if (xyz.x() < m_dim.x() - 2)
				res += m_data[getVoxelIndexUnsigned(m_dim, xyz.x() + 1, xyz.y(), xyz.z())];

			if (xyz.y() > 0)
				res += m_data[getVoxelIndexUnsigned(m_dim, xyz.x(), xyz.y() - 1, xyz.z())];

			if (xyz.y() < m_dim.y() - 2)
				res += m_data[getVoxelIndexUnsigned(m_dim, xyz.x(), xyz.y() + 1, xyz.z())];

			if (xyz.z() > 0)
				res += m_data[getVoxelIndexUnsigned(m_dim, xyz.x(), xyz.y(), xyz.z() - 1)];

			if (xyz.z() < m_dim.z() - 2)
				res += m_data[getVoxelIndexUnsigned(m_dim, xyz.x(), xyz.y(), xyz.z() + 1)];

			res /= 6;
			return 1 - res;
	}

		thrust::device_vector<uint32_t> culled_filter(const thrust::device_vector<uint32_t>& filter, const Vector3ui& dim)
		{
			thrust::device_vector<uint32_t> iterim1(filter.size());

			thrust::transform(
				thrust::counting_iterator<uint32_t>(0),
				thrust::counting_iterator<uint32_t>(filter.size()),
				iterim1.begin(),
				CullHiddenVoxels(filter.data().get(), dim));

			return iterim1;
		}

		__host__ __device__
			GatherCompacted::GatherCompacted(Vector3ui* data, Vector3ui dim)
			: m_dim(std::move(dim)), m_data(data)
		{}

		__host__ __device__
			void GatherCompacted::operator()(const thrust::tuple<uint32_t, uint32_t, uint32_t>& input)
		{
			if (thrust::get<0>(input) == 0)
				return;

			m_data[thrust::get<1>(input) - 1] = indexToXYZ(thrust::get<2>(input), m_dim);
		}

		std::vector<Vector3ui> extract_visual_voxels(const thrust::device_vector<BitVectorVoxel>& in, const Vector3ui& dim)
		{
			const size_t& in_size = in.size();

			thrust::device_vector<uint32_t> occupation(in_size);
			thrust::device_vector<uint32_t> culled(in_size);

			thrust::transform(
				in.begin(), in.end(),
				occupation.begin(), OccupiedVoxels());

			thrust::transform(
				thrust::counting_iterator<uint32_t>(0),
				thrust::counting_iterator<uint32_t>(in_size),
				culled.begin(),
				CullHiddenVoxels(occupation.data().get(), dim));

			thrust::inclusive_scan(culled.begin(), culled.end(), occupation.begin());
			const size_t new_size = occupation.back();

			thrust::device_vector<Vector3ui> result(new_size);

			thrust::for_each(
				thrust::make_zip_iterator(culled.begin(), occupation.begin(), thrust::counting_iterator<uint32_t>(0)),
				thrust::make_zip_iterator(culled.end(), occupation.end(), thrust::counting_iterator<uint32_t>(in_size)),
				GatherCompacted(result.data().get(), dim)
			);

			std::vector<Vector3ui> final_res(new_size);
			thrust::copy(result.begin(), result.end(), final_res.begin());

			return final_res;
		}

		template<> std::vector<Vector3ui> extract_visual_voxels(const TemplateVoxelMap<BitVoxel<BIT_VECTOR_LENGTH>>& in);
		template<> std::vector<Vector3ui> extract_visual_voxels(const TemplateVoxelMap<ProbabilisticVoxel>& in);
		template<> std::vector<Vector3ui> extract_visual_voxels(const TemplateVoxelMap<CountingVoxel>& in);
		template<> std::vector<Vector3ui> extract_visual_voxels(const TemplateVoxelMap<DistanceVoxel>& in);
	} // end of namespace voxelmap
} // end of namespace voxellist

#ifdef LOCAL_DEBUG
#undef LOCAL_DEBUG
#endif