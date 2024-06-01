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
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
 //----------------------------------------------------------------------
#ifndef ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_HPP_INCLUDED
#define ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_HPP_INCLUDED

#include "VoxelMapOperations.h"
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/DistanceVoxel.hpp>

#include "VoxelMapOperationsPBA.hpp"


#if defined(__INTELLISENSE___) || defined(__RESHARPER__) 
// in here put whatever is your favorite flavor of intellisense workarounds
#ifndef __CUDACC__ 
#define __CUDACC__
#include <device_functions.h>
#include "device_launch_parameters.h"
#endif
#endif

namespace gpu_voxels {
	namespace voxelmap {

		template<class Voxel>
		__global__
		void kernelClearVoxelMap(Voxel* voxelmap, const uint32_t voxelmap_size)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
				voxelmap[i] = Voxel();
		}

		template<std::size_t bit_length>
		__global__
		void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, const uint32_t voxelmap_size, const uint32_t bit_index)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
			{
				BitVector<bit_length>& bit_vector = voxelmap[i].bitVector();
				if (bit_vector.getBit(bit_index))
					bit_vector.clearBit(bit_index);
			}
		}

		template<std::size_t bit_length>
		__global__
		void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, uint32_t voxelmap_size,
			BitVector<bit_length> bits)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
			{
				BitVector<bit_length>& bit_vector = voxelmap[i].bitVector();
				if (!(bit_vector & bits).isZero())
				{
					BitVector<bit_length> tmp = bit_vector;
					tmp = tmp & ~bits;
					bit_vector = tmp;
				}
			}
		}

		/*! Collide two voxel maps.
		 * Voxels are considered occupied for values
		 * greater or equal given thresholds.
		 */
		template<class Voxel, class OtherVoxel, class Collider>
		__global__
		void kernelCollideVoxelMaps(Voxel* voxelmap, const uint32_t voxelmap_size, OtherVoxel* other_map,
			Collider collider, bool* results)
		{
			__shared__ bool cache[cMAX_THREADS_PER_BLOCK];
			uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			uint32_t cache_index = threadIdx.x;
			cache[cache_index] = false;
			bool temp = false;

			while (i < voxelmap_size)
			{
				if (collider.collide(voxelmap[i], other_map[i]))
					temp = true;
				i += blockDim.x * gridDim.x;
			}

			cache[cache_index] = temp;
			__syncthreads();

			uint32_t j = blockDim.x / 2;

			while (j != 0)
			{
				if (cache_index < j)
					cache[cache_index] = cache[cache_index] || cache[cache_index + j];

				__syncthreads();
				j /= 2;
			}

			// copy results from this block to global memory
			if (cache_index == 0)
			{
				//    // FOR MEASUREMENT TEMPORARILY EDITED:
				//    results[blockIdx.x] = true;
				results[blockIdx.x] = cache[0];
			}
		}

		/* Collide two voxel maps with storing collision info (for debugging only)
		 * Voxels are considered occupied for values
		 * greater or equal given thresholds.
		 *
		 * Collision info is stored within eBVM_COLLISION model for 'other_map'.
		 * Warning: Original model is modified!
		 */
		template<class Voxel, class OtherVoxel, class Collider>
		__global__
		void kernelCollideVoxelMapsDebug(Voxel* voxelmap, const uint32_t voxelmap_size, const OtherVoxel* other_map,
			Collider collider, uint16_t* results)
		{
			//#define DISABLE_STORING_OF_COLLISIONS
			__shared__ uint16_t cache[cMAX_THREADS_PER_BLOCK];
			uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			uint32_t cache_index = threadIdx.x;
			cache[cache_index] = 0;

			while (i < voxelmap_size)
			{
				// todo / note: at the moment collision check is only used for DYNAMIC and SWEPT VOLUME meaning, static is used for debugging
				const bool collision = collider.collide(voxelmap[i], other_map[i]);
				if (collision) // store collision info
				{
#ifndef DISABLE_STORING_OF_COLLISIONS
					//      other_map[i].occupancy = 255;
					//      other_map[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
					voxelmap[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
#endif
					cache[cache_index] += 1;
				}
				i += blockDim.x * gridDim.x;
			}

			// debug: print collision coordinates

		  //  if (temp)
		  //  {
		  //    Vector3ui col_coord = mapToVoxels(voxelmap, dimensions, &(voxelmap[i]));
		  //    printf("Collision at voxel (%u) = (%u, %u, %u). Memory addresses are %p and %p.\n",
		  //           i, col_coord.x, col_coord.y, col_coord.z, (void*)&(voxelmap[i]), (void*)&(other_map[i]));
		  //  }
			__syncthreads();

			uint32_t j = blockDim.x / 2;

			while (j != 0)
			{
				if (cache_index < j)
					cache[cache_index] = cache[cache_index] + cache[cache_index + j];

				__syncthreads();
				j /= 2;
			}

			// copy results from this block to global memory
			if (cache_index == 0)
				results[blockIdx.x] = cache[0];
#undef DISABLE_STORING_OF_COLLISIONS
		}


		template<std::size_t length, class OtherVoxel, class Collider>
		__global__
		void kernelCollideVoxelMapsBitvector(BitVoxel<length>* voxelmap, const uint32_t voxelmap_size,
			const OtherVoxel* other_map, Collider collider,
			BitVector<length>* results, uint16_t* num_collisions, const uint16_t sv_offset)
		{
			extern __shared__ BitVector<length> cache[]; //[cMAX_THREADS_PER_BLOCK];
			__shared__ uint16_t cache_num[cMAX_THREADS_PER_BLOCK];
			uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			uint32_t cache_index = threadIdx.x;
			cache[cache_index] = BitVector<length>();
			cache_num[cache_index] = 0;
			BitVector<length> temp;

			while (i < voxelmap_size)
			{
				const bool collision = collider.collide(voxelmap[i], other_map[i], &temp, sv_offset);
				if (collision) // store collision info
				{
#ifndef DISABLE_STORING_OF_COLLISIONS
					//      other_map[i].occupancy = 255;
					//      other_map[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
					voxelmap[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
#endif
					cache[cache_index] = temp;
					cache_num[cache_index] += 1;
				}
				i += blockDim.x * gridDim.x;
			}
			__syncthreads();

			uint32_t j = blockDim.x / 2;

			while (j != 0)
			{
				if (cache_index < j)
				{
					cache[cache_index] = cache[cache_index] | cache[cache_index + j];
					cache_num[cache_index] = cache_num[cache_index] + cache_num[cache_index + j];
				}
				__syncthreads();
				j /= 2;
			}

			// copy results from this block to global memory
			if (cache_index == 0)
			{
				// FOR MEASUREMENT TEMPORARILY EDITED:
				//results[blockIdx.x] = true;
				results[blockIdx.x] = cache[0];
				num_collisions[blockIdx.x] = cache_num[0];
			}
		}

		template<class Voxel>
		__global__
		void kernelInsertGlobalPointCloud(Voxel* voxelmap, Vector3ui map_dim, float voxel_side_length,
			const Vector3f* points, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
			bool* points_outside_map)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
			{
				const Vector3ui uint_coords = mapToVoxels(voxel_side_length, points[i]);
				//check if point is in the range of the voxel map
				if (uint_coords.x() < map_dim.x() && uint_coords.y() < map_dim.y()
					&& uint_coords.z() < map_dim.z())
				{
					Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(map_dim, uint_coords)];
					voxel->insert(voxel_meaning);
				}
				else
				{
					if (points_outside_map) *points_outside_map = true;
					//       printf("Point (%u,%u,%u) is not in the range of the voxel map \n", points[i].x, points[i].y,
					//              points[i].z);
				}
			}
		}

		template<class Voxel>
		__global__
		void kernelInsertCoordinateTuples(Voxel* voxelmap, Vector3ui map_dim, float voxel_side_length,
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
					Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(map_dim, uint_coords)];
					voxel->insert(voxel_meaning);
				}
				else
				{
					if (points_outside_map) *points_outside_map = true;
					//       printf("Point (%u,%u,%u) is not in the range of the voxel map \n", points[i].x, points[i].y,
					//              points[i].z);
				}
			}
		}

		template<class Voxel>
		__global__
		void kernelInsertDilatedCoordinateTuples(Voxel* voxelmap, Vector3ui dimensions,
			const Vector3ui* coordinates, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
			bool* points_outside_map)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
			{
				const Vector3ui uint_center_coords = coordinates[i];
				// Check if center voxel is in range of the voxel map
				if (uint_center_coords.x() >= dimensions.x() || uint_center_coords.y() >= dimensions.y() || uint_center_coords.z() >= dimensions.z())
				{
					if (points_outside_map) *points_outside_map = true;
					continue;
				}

				constexpr int32_t SE_SIZE = 1;
				// Iterate neighbors
				for (int32_t x = -SE_SIZE; x <= SE_SIZE; x++)
				{
					for (int32_t y = -SE_SIZE; y <= SE_SIZE; y++)
					{
						for (int32_t z = -SE_SIZE; z <= SE_SIZE; z++)
						{
							Vector3i int_neighbor_coords = uint_center_coords.cast<int32_t>() + Vector3i(x, y, z);
							// Check if neighbor voxel is in range of the voxel map
							if (int_neighbor_coords.x() < dimensions.x() && int_neighbor_coords.y() < dimensions.y() && int_neighbor_coords.z() < dimensions.z()
								&& int_neighbor_coords.x() >= 0 && int_neighbor_coords.y() >= 0 && int_neighbor_coords.z() >= 0)
							{
								Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, Vector3ui((uint32_t)int_neighbor_coords.x(), (uint32_t)int_neighbor_coords.y(), (uint32_t)int_neighbor_coords.z()))];
								voxel->insert(voxel_meaning);
							}
						}
					}
				}
			}
		}

		template<class Voxel>
		__global__
		void kernelErode(Voxel* voxelmap_out, const Voxel* voxelmap_in, Vector3ui dimensions, float occupied_threshold, float erode_threshold)
		{
			constexpr int32_t SE_SIZE = 1;
			const Vector3ui uint_center_coords(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
			if (uint_center_coords.x() >= dimensions.x() || uint_center_coords.y() >= dimensions.y() || uint_center_coords.z() >= dimensions.z())
				return;

			// Count number of occupied neighbors, and total number of neighbors (might be less that 27 at map borders)
			uint32_t total = 0;
			uint32_t occupied = 0;
			for (int32_t x = -SE_SIZE; x <= SE_SIZE; x++)
			{
				for (int32_t y = -SE_SIZE; y <= SE_SIZE; y++)
				{
					for (int32_t z = -SE_SIZE; z <= SE_SIZE; z++)
					{
						const Vector3i int_neighbor_coords = uint_center_coords.cast<int32_t>() + Vector3i(x, y, z);
						// Check if neighbor voxel is in range of the voxel map, and is not the center voxel
						if (int_neighbor_coords.x() < dimensions.x() && int_neighbor_coords.y() < dimensions.y() && int_neighbor_coords.z() < dimensions.z()
							&& int_neighbor_coords.x() >= 0 && int_neighbor_coords.y() >= 0 && int_neighbor_coords.z() >= 0
							&& (x != 0 || y != 0 || z != 0))
						{
							total++;
							const Voxel& neighbor_voxel = voxelmap_in[getVoxelIndexUnsigned(dimensions, Vector3ui((uint32_t)int_neighbor_coords.x(), (uint32_t)int_neighbor_coords.y(), (uint32_t)int_neighbor_coords.z()))];
							if (neighbor_voxel.isOccupied(occupied_threshold))
							{
								occupied++;
							}
						}
					}
				}
			}

			Voxel& voxel_out = voxelmap_out[getVoxelIndexUnsigned(dimensions, uint_center_coords)];
			if (static_cast<float>(occupied) / total < erode_threshold)
			{
				// Clear voxel
				voxel_out = Voxel();
			}
			else
			{
				// Keep voxel
				voxel_out = voxelmap_in[getVoxelIndexUnsigned(dimensions, uint_center_coords)];
			}
		}

		template<class Voxel>
		__global__
		void kernelInsertMetaPointCloud(Voxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			BitVoxelMeaning voxel_meaning, Vector3ui dimensions, float voxel_side_length,
			bool* points_outside_map)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
				i += blockDim.x * gridDim.x)
			{
				const Vector3ui uint_coords = mapToVoxels(voxel_side_length,
					meta_point_cloud->clouds_base_addresses[0][i]);

				//        printf("Point @(%f,%f,%f)\n",
				//               meta_point_cloud->clouds_base_addresses[0][i].x,
				//               meta_point_cloud->clouds_base_addresses[0][i].y,
				//               meta_point_cloud->clouds_base_addresses[0][i].z);

					//check if point is in the range of the voxel map
				if (uint_coords.x() < dimensions.x() && uint_coords.y() < dimensions.y()
					&& uint_coords.z() < dimensions.z())
				{
					Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, uint_coords)];
					voxel->insert(voxel_meaning);

					//        printf("Inserted Point @(%u,%u,%u) into the voxel map \n",
					//               integer_coordinates.x,
					//               integer_coordinates.y,
					//               integer_coordinates.z);

				}
				else
				{
					if (points_outside_map) *points_outside_map = true;
					//       printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
					//              meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
					//              meta_point_cloud->clouds_base_addresses[0][i].z);
				}
			}
		}

		//TODO: specialize every occurence of voxel->insert(meaning) for DistanceVoxel to use voxel->insert(integer_coordinates, meaning)

		template<class Voxel>
		__global__
		void kernelInsertMetaPointCloud(Voxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			BitVoxelMeaning* voxel_meanings, const Vector3ui map_dim,
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


				const Vector3ui uint_coords = mapToVoxels(voxel_side_length,
					meta_point_cloud->clouds_base_addresses[0][i]);

				//        printf("Point @(%f,%f,%f)\n",
				//               meta_point_cloud->clouds_base_addresses[0][i].x,
				//               meta_point_cloud->clouds_base_addresses[0][i].y,
				//               meta_point_cloud->clouds_base_addresses[0][i].z);

					//check if point is in the range of the voxel map
				if (uint_coords.x() < map_dim.x() && uint_coords.y() < map_dim.y()
					&& uint_coords.z() < map_dim.z())
				{
					Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(map_dim, uint_coords)];
					voxel->insert(voxel_meanings[sub_cloud]);

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
			}
		}

		//template<std::size_t length>
		//__global__
		//void kernelInsertSensorDataWithRayCasting(ProbabilisticVoxel* voxelmap, const uint32_t voxelmap_size,
		//                                          const Vector3ui dimensions, const float voxel_side_length,
		//                                          Sensor* sensor, const Vector3f* sensor_data,
		//                                          const bool cut_real_robot, BitVoxel<length>* robotmap,
		//                                          const uint32_t bit_index)
		//{
		//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < voxelmap_size) && (i < sensor->data_size);
		//      i += gridDim.x * blockDim.x)
		//  {
		//    if (!(isnan(sensor_data[i].x) || isnan(sensor_data[i].y) || isnan(sensor_data[i].z)))
		//    {
		//      const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, sensor_data[i]);
		//      const Vector3ui sensor_coordinates = mapToVoxels(voxel_side_length, sensor->position);
		//
		//      /* both data and sensor coordinates must
		//       be within boundaries for raycasting to work */
		//      if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
		//          && (integer_coordinates.z < dimensions.z) && (sensor_coordinates.x < dimensions.x)
		//          && (sensor_coordinates.y < dimensions.y) && (sensor_coordinates.z < dimensions.z))
		//      {
		//        bool update = false;
		//        if (robotmap && cut_real_robot)
		//        {
		//          BitVoxel<length>* robot_voxel = getVoxelPtr(robotmap, dimensions, integer_coordinates.x,
		//                                                      integer_coordinates.y, integer_coordinates.z);
		//
		////          if (!((robot_voxel->occupancy > 0) && (robot_voxel->voxelmeaning == eBVM_OCCUPIED))) // not occupied by robot
		////           {
		//          update = !robot_voxel->bitVector().getBit(bit_index); // not occupied by robot
		////          else // else: sensor sees robot, no need to insert data.
		////          {
		////            printf("cutting robot from sensor data in kernel %u\n", i);
		////          }
		//        }
		//        else
		//          update = true;
		//
		//        if (update)
		//        {
		//          // sensor does not see robot, so insert data into voxelmap
		//          // raycasting
		//          rayCast(voxelmap, dimensions, sensor, sensor_coordinates, integer_coordinates);
		//
		//          // insert measured data itself:
		//          ProbabilisticVoxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x,
		//                                                  integer_coordinates.y, integer_coordinates.z);
		//          voxel->updateOccupancy(cSENSOR_MODEL_OCCUPIED);
		////            voxel->voxelmeaning = eBVM_OCCUPIED;
		////            increaseOccupancy(voxel, cSENSOR_MODEL_OCCUPIED); // todo: replace with "occupied" of sensor model
		//        }
		//      }
		//    }
		//  }
		//}

		/* Insert sensor data into voxel map.
		 * Assumes sensor data is already transformed
		 * into world coordinate system.
		 * If cut_real_robot is enabled one has to
		 * specify pointer to the robot voxel map.
		 * The robot voxels will be assumed 100% certain
		 * and cut from sensor data.
		 * See also function with ray casting.
		 */
		template<std::size_t length, class RayCasting>
		__global__
			void kernelInsertSensorData(ProbabilisticVoxel* voxelmap, const uint32_t voxelmap_size,
				const Vector3ui dimensions, const float voxel_side_length, const Vector3f sensor_pose,
				const Vector3f* sensor_data, const size_t num_points, const bool cut_real_robot,
				BitVoxel<length>* robotmap, const uint32_t bit_index, RayCasting rayCaster)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size && i < num_points;
				i += gridDim.x * blockDim.x)
			{
				if (!(isnan(sensor_data[i].x()) || isnan(sensor_data[i].y()) || isnan(sensor_data[i].z())))
				{
					const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, sensor_data[i]);
					const Vector3ui sensor_coordinates = mapToVoxels(voxel_side_length, sensor_pose);

					/* both data and sensor coordinates must
					 be within boundaries for raycasting to work */
					if (integer_coordinates.x() < dimensions.x() && integer_coordinates.y() < dimensions.y()
						&& integer_coordinates.z() < dimensions.z() && sensor_coordinates.x() < dimensions.x()
						&& sensor_coordinates.y() < dimensions.y() && sensor_coordinates.z() < dimensions.z())
					{
						bool update = false;
						if (cut_real_robot)
						{
							BitVoxel<length>* robot_voxel = getVoxelPtr(robotmap, dimensions, integer_coordinates.x(),
								integer_coordinates.y(), integer_coordinates.z());

							update = !robot_voxel->bitVector().getBit(bit_index); // not occupied by robot
						}
						else
							update = true;

						if (update)
						{
							// sensor does not see robot, so insert data into voxelmap
							// raycasting
							rayCaster.rayCast(voxelmap, dimensions, sensor_coordinates, integer_coordinates);

							// insert measured data itself afterwards, so it overrides free voxels from raycaster:
							ProbabilisticVoxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x(),
								integer_coordinates.y(), integer_coordinates.z());
							voxel->updateOccupancy(cSENSOR_MODEL_OCCUPIED);
						}
					}
				}
			}
		}

		template<std::size_t length>
		__global__
			void kernelShiftBitVector(BitVoxel<length>* voxelmap,
				const uint32_t voxelmap_size, uint8_t shift_size)
		{
			for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
			{
				performLeftShift(voxelmap[i].bitVector(), shift_size);
			}
		}

		template <class Voxel>
		std::vector<Vector3ui> extract_visual_voxels(const TemplateVoxelMap<Voxel>& in)
		{
			const auto& device_vector = in.getConstDeviceDataPtr();
			return extract_visual_voxels(device_vector, in.getDimensions());
		}
	} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif