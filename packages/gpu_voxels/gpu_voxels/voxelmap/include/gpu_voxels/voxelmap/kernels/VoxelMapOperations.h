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
#ifndef ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_H_INCLUDED
#define ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_H_INCLUDED

#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>
#include <gpu_voxels/voxel/DistanceVoxel.h>
#include <thrust/device_vector.h>

#include "VoxelMapOperationsPBA.h"

namespace gpu_voxels
{
	namespace voxelmap
	{
		/* ------------------ DEVICE FUNCTIONS ------------------ */

		// VoxelMap addressing
		//! Maps 3D voxel coordinates to linear voxel index
		__device__ __host__     __forceinline__
			uint32_t getVoxelIndexUnsigned(const Vector3ui& dimensions,
				const uint32_t x, const uint32_t y, const uint32_t z)
		{
			return z * dimensions.x() * dimensions.y() + y * dimensions.x() + x;
		}

		//! Maps 3D voxel coordinates to linear voxel index
		__host__ __device__      __forceinline__
			uint32_t getVoxelIndexUnsigned(const Vector3ui& dimensions, const Vector3ui& coords)
		{
			return coords.z() * dimensions.x() * dimensions.y() + coords.y() * dimensions.x() + coords.x();
		}

		//! Maps 3D voxel coordinates to linear voxel index
		__device__ __host__     __forceinline__
			int32_t getVoxelIndexSigned(const Vector3ui& dimensions,
				const int32_t x, const int32_t y, const int32_t z)
		{
			return z * (int32_t)dimensions.x() * (int32_t)dimensions.y() + y * (int32_t)dimensions.x() + x;
		}

		//! Maps 3D voxel coordinates to linear voxel index
		__host__ __device__      __forceinline__
			int32_t getVoxelIndexSigned(const Vector3ui& dimensions, const Vector3i& offset)
		{
			// cast the values to prevent underflow
			return offset.z() * (int32_t)dimensions.x() * (int32_t)dimensions.y() + offset.y() * (int32_t)dimensions.x() + offset.x();
		}

		template<class Voxel>
		__device__ __host__     __forceinline__
		Voxel getVoxel(const thrust::device_vector<Voxel>& voxelmap, const Vector3ui& dimensions,
			const uint32_t x, const uint32_t y, const uint32_t z)
		{
			return voxelmap[getVoxelIndexUnsigned(dimensions, x, y, z)];
		}

		template<class Voxel>
		__device__ __host__     __forceinline__
		Voxel* getVoxelPtr(const Voxel* voxelmap, const Vector3ui& dimensions,
			const uint32_t x, const uint32_t y, const uint32_t z)
		{
			return (Voxel*)(voxelmap + getVoxelIndexUnsigned(dimensions, x, y, z));
		}

		template<class Voxel>
		__device__ __host__     __forceinline__
			Voxel* getVoxelPtr(const Voxel* voxelmap, const Vector3ui& dimensions,
				const Vector3ui& voxel_coords)
		{
			return (Voxel*)(voxelmap + getVoxelIndexUnsigned(dimensions, voxel_coords));
		}

		template<class Voxel>
		__device__ __host__     __forceinline__
			Voxel* getVoxelPtrSignedOffset(const Voxel* voxelmap, const Vector3ui& dimensions,
				const int32_t x, const int32_t y, const int32_t z)
		{
			return (Voxel*)(voxelmap + getVoxelIndexSigned(dimensions, x, y, z));
		}

		template<class Voxel>
		__device__ __host__     __forceinline__
			Voxel* getVoxelPtrSignedOffset(const Voxel* voxelmap, const Vector3ui& dimensions,
				const Vector3i& voxel_offset)
		{
			return (Voxel*)(voxelmap + getVoxelIndexSigned(dimensions, voxel_offset));
		}

		__host__ __device__
		inline Vector3ui indexToXYZ(uint32_t index, const Vector3ui& dim)
		{
			Vector3ui r;
			r.z() = index / (dim.x() * dim.y());
			r.y() = index / dim.x() - dim.y() * r.z();
			r.x() = index - dim.x() * dim.y() * r.z() - dim.x() * r.y();
			return r;
		}

		//! Maps a voxel address to discrete voxel coordinates
		template<class Voxel>
		__device__ __host__     __forceinline__
			Vector3ui mapToVoxels(const Voxel* voxelmap, const Vector3ui& dimensions,
				const Voxel* voxel)
		{
			Vector3ui uint_coords;
			int voxel_index = voxel - voxelmap;
			uint_coords.z() = voxel_index / (dimensions.x() * dimensions.y());
			uint_coords.y() = (voxel_index -= uint_coords.z() * (dimensions.x() * dimensions.y())) / dimensions.x();
			uint_coords.x() = voxel_index -= uint_coords.y() * dimensions.x();
			return uint_coords;
		}

		//! Partitioning of continuous data into voxels. Maps float coordinates to dicrete voxel coordinates.
		__device__ __host__     __forceinline__
			Vector3ui mapToVoxels(const float voxel_side_length, const Vector3f& coordinates)
		{
			Vector3ui uint_coords;
			uint_coords.x() = static_cast<uint32_t>(floor(coordinates.x() / voxel_side_length));
			uint_coords.y() = static_cast<uint32_t>(floor(coordinates.y() / voxel_side_length));
			uint_coords.z() = static_cast<uint32_t>(floor(coordinates.z() / voxel_side_length));

			// printf("coordinates.x = %f -> uint_coords.x = %u \n", coordinates.x, uint_coords.x);
			return uint_coords;
		}

		//! Partitioning of continuous data into voxels. Maps float coordinates to dicrete voxel coordinates.
		__device__ __host__     __forceinline__
			Vector3i mapToVoxelsSigned(const float voxel_side_length, const Vector3f& coordinates)
		{
			Vector3i integer_coordinates;
			integer_coordinates.x() = static_cast<int32_t>(floor(coordinates.x() / voxel_side_length));
			integer_coordinates.y() = static_cast<int32_t>(floor(coordinates.y() / voxel_side_length));
			integer_coordinates.z() = static_cast<int32_t>(floor(coordinates.z() / voxel_side_length));

			// printf("coordinates.x = %f -> integer_coordinates.x = %u \n", coordinates.x, integer_coordinates.x);
			return integer_coordinates;
		}

		//! Maps a voxel address to discrete voxel coordinates
		__device__ __host__ __forceinline__
			Vector3i mapToVoxelsSigned(int linear_id, const Vector3ui& dimensions)
		{
			Vector3i integer_coordinates;
			integer_coordinates.z() = linear_id / (dimensions.x() * dimensions.y());
			integer_coordinates.y() = (linear_id -= integer_coordinates.z() * (dimensions.x() * dimensions.y())) / dimensions.x();
			integer_coordinates.x() = linear_id -= integer_coordinates.y() * dimensions.x();
			return integer_coordinates;
		}

		__device__ __host__ __forceinline__
			Vector3ui mapToVoxels(unsigned int linear_id, const Vector3ui& dimensions)
		{
			Vector3ui integer_coordinates;
			integer_coordinates.z() = linear_id / (dimensions.x() * dimensions.y());
			integer_coordinates.y() = (linear_id -= integer_coordinates.z() * (dimensions.x() * dimensions.y())) / dimensions.x();
			integer_coordinates.x() = linear_id -= integer_coordinates.y() * dimensions.x();
			return integer_coordinates;
		}

		template<class Voxel>
		__device__ __host__ __forceinline__
		Voxel* getHighestVoxelPtr(const Voxel* base_addr, const Vector3ui& dimensions)
		{
			return getVoxelPtr(base_addr, dimensions, dimensions.x() - 1, dimensions.y() - 1, dimensions.z() - 1);
		}

		//! Returns the center of a voxel as float coordinates. Mainly for boost test purposes!
		__device__ __host__ __forceinline__
			Vector3f getVoxelCenter(float voxel_side_length, const Vector3ui& voxel_coords)
		{

			return {
				voxel_coords.x()* voxel_side_length + voxel_side_length / 2.f,
				voxel_coords.y()* voxel_side_length + voxel_side_length / 2.f,
				voxel_coords.z()* voxel_side_length + voxel_side_length / 2.f
			};
		}

		//! update min_voxel if newVoxel is valid and closer
		__device__ __forceinline__
			void updateMinVoxel(const DistanceVoxel& new_voxel, DistanceVoxel& min_voxel, const Vector3i& cur_pos)
		{
			// optimize: don't skip obstacles in global function to reduce divergence? remove check for 0 or -1 here

			  //TODO optimise: pass min_distance as argument; return new min_distance; most calls of updateMinVoxel will just read and not change the value
			  //TODO optimise: change function updateMinVoxel to macro? check performance
			const int min_distance = min_voxel.squaredObstacleDistance(cur_pos);

			// check if min_distance is invalid, or just ignore
			// assumption: p_min_distance is either MAX_DISTANCE or a valid distance > 0
			if (min_distance == PBA_OBSTACLE_DISTANCE) { //minVoxel uninitialised, not an obstacle
				printf("should never happen: updateVoxel min_distance: %i (pos: %i/%i/%i)\n", min_distance, cur_pos.x(), cur_pos.y(), cur_pos.z());
				return;
			}
			else { //TODO:: check if intermediate result is too large
				const Vector3ui new_obstacle_u = new_voxel.getObstacle();
				const Vector3i new_obstacle = new_obstacle_u.cast<int32_t>();

				const int32_t new_distance = (cur_pos - new_obstacle).array().square().sum(); //squared distance

				// int curiosity: other_map[0] 2147483647/2147483647/2147483647 (distance: 3)
				if (new_obstacle.x() != PBA_UNINITIALISED_COORD
					&& new_distance < min_distance
					&& new_obstacle.y() != PBA_UNINITIALISED_COORD
					&& new_obstacle.z() != PBA_UNINITIALISED_COORD) { //need to update; never overwrite with UNINITIALIZED
					min_voxel = DistanceVoxel(new_obstacle_u);
				} // else: don't need to consider new_voxel
			}
		}

		struct RayCaster
		{
		public:

			//! raycasting from one point to another marks decreases voxel occupancy along the ray
			__device__ __forceinline__
				void rayCast(ProbabilisticVoxel* voxelmap, const Vector3ui& dimensions,
					const Vector3ui& from, const Vector3ui& to)
			{
				int32_t difference_x = 0;
				int32_t difference_y = 0;
				int32_t difference_z = 0;

				int32_t x = 0;
				int32_t y = 0;
				int32_t z = 0;

				int32_t n;

				int8_t x_increment = 0;
				int8_t y_increment = 0;
				int8_t z_increment = 0;

				int32_t error_xy = 0;
				int32_t error_xz = 0;
				int32_t error_yz = 0;
				// -------------------------------------------  //

				// differences
				if (to.x() > from.x())
				{
					difference_x = to.x() - from.x();
					x_increment = 1;
				}
				else
				{
					difference_x = from.x() - to.x();
					x_increment = -1;
				}

				if (to.y() > from.y())
				{
					difference_y = to.y() - from.y();
					y_increment = 1;
				}
				else
				{
					difference_y = from.y() - to.y();
					y_increment = -1;
				}

				if (to.z() > from.z())
				{
					difference_z = to.z() - from.z();
					z_increment = 1;
				}
				else
				{
					difference_z = from.z() - to.z();
					z_increment = -1;
				}

				// pointer to currently visited voxel
				ProbabilisticVoxel* voxel = nullptr;

				// start values
				x = from.x();
				y = from.y();
				z = from.z();

				// number of cells to visit
				n = 1 + difference_x + difference_y + difference_z;

				// error between x- and y- difference
				error_xy = difference_x - difference_y;

				// error between x- and z- difference
				error_xz = difference_x - difference_z;

				// error between y- and z- difference
				error_yz = difference_y - difference_z;

				// double differences to avoid float values
				difference_x *= 2;
				difference_y *= 2;
				difference_z *= 2;

				for (; n > 0; n--)
				{
					//printf("visiting cell (%d, %d, %d)\n", x, y, z);
					voxel = getVoxelPtr(voxelmap, dimensions, x, y, z);
					//voxel->voxelmeaning = eBVM_OCCUPIED;

					voxel->updateOccupancy(cSENSOR_MODEL_FREE);

					//decreaseOccupancy(voxel, cSENSOR_MODEL_FREE); // todo: replace with "free" of sensor model


					if (error_xy > 0 && error_xz > 0)
					{
						// walk in x direction until error_xy or error_xz is below 0
						x += x_increment;
						error_xy -= difference_y;
						error_xz -= difference_z;
					}
					else
					{
						if (error_yz > 0)
						{
							// walk in y direction
							y += y_increment;
							error_xy += difference_x;
							error_yz -= difference_z;
						}
						else
						{
							// walk in z direction
							z += z_increment;
							error_xz += difference_x;
							error_yz += difference_y;
						}
					}
				}
			}
		};

		struct DummyRayCaster : public RayCaster
		{
		public:
			__device__ __forceinline__
				void rayCast(ProbabilisticVoxel* voxelmap, const Vector3ui& dimensions,
					const Vector3ui& from, const Vector3ui& to)
			{
				// override and do nothing
			}
		};

		///* ------------------ KERNELS ------------------ */

		//! Clear voxel occupancy
		template<class Voxel>
		__global__
			void kernelClearVoxelMap(Voxel* voxelmap, uint32_t voxelmap_size);

		//! Clear voxel occupancy for specific voxel_meaning
		template<std::size_t bit_length>
		__global__
			void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, uint32_t voxelmap_size, uint32_t bit_index);

		template<std::size_t bit_length>
		__global__
			void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, uint32_t voxelmap_size,
				BitVector<bit_length> bits);

		///*! Print voxel info from within kernel.
		// */
		//__global__
		//void kernelDumpVoxelMap(const Voxel* voxelmap, const Vector3ui dimensions, const uint32_t voxelmap_size);


		/*! Insert sensor data into voxel map.
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
			void kernelInsertSensorData(ProbabilisticVoxel* voxelmap, uint32_t voxelmap_size,
			                            Vector3ui dimensions, float voxel_side_length, Vector3f sensor_pose,
				const Vector3f* sensor_data, size_t num_points, bool cut_real_robot,
				BitVoxel<length>* robotmap, uint32_t bit_index, RayCasting rayCaster);

		/*!
		 * Collide two voxel maps.
		 */
		template<class Voxel, class OtherVoxel, class Collider>
		__global__
			void kernelCollideVoxelMaps(Voxel* voxelmap, uint32_t voxelmap_size, OtherVoxel* other_map,
				Collider collider, bool* results);



		/*!
		 * Collide two Voxelmaps but test against the Bitvector of the other voxelmap
		 *
		 */
		template<std::size_t length, class OtherVoxel, class Collider>
		__global__
			void kernelCollideVoxelMapsBitvector(BitVoxel<length>* voxelmap, uint32_t voxelmap_size,
				const OtherVoxel* other_map, Collider collider,
				BitVector<length>* results, uint16_t* num_collisions, uint16_t sv_offset);


		/*! Collide two voxel maps with storing collision info (for debugging only)
		 * Voxels are considered occupied for values
		 * greater or equal given thresholds.
		 *
		 * Collision info is stored within static model for 'other_map'.
		 * Warning: Original model is modified!
		 */
		template<class Voxel, class OtherVoxel, class Collider>
		__global__
		void kernelCollideVoxelMapsDebug(Voxel* voxelmap, uint32_t voxelmap_size, const OtherVoxel* other_map,
			Collider collider, uint16_t* results);

		/*!
		 * Inserts pointcloud with global coordinates.
		 *
		 */
		template<class Voxel>
		__global__
		void kernelInsertGlobalPointCloud(Voxel* voxelmap, Vector3ui map_dim, float voxel_side_length,
			const Vector3f* points, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
			bool* points_outside_map);

		template<>
		__global__
		void kernelInsertGlobalPointCloud(DistanceVoxel* voxelmap, Vector3ui map_dim, float voxel_side_length,
			const Vector3f* points, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
			bool* points_outside_map);

		template<class Voxel>
		__global__
			void kernelInsertCoordinateTuples(Voxel* voxelmap, Vector3ui map_dim, float voxel_side_length,
				const Vector3ui* coordinates, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
				bool* points_outside_map);

		template<>
		__global__
		void kernelInsertCoordinateTuples(DistanceVoxel* voxelmap, Vector3ui map_dim, float voxel_side_length,
			const Vector3ui* coordinates, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
			bool* points_outside_map);

		template<class Voxel>
		__global__
		void kernelInsertDilatedCoordinateTuples(Voxel* voxelmap, Vector3ui dimensions,
			const Vector3ui* coordinates, std::size_t sizePoints, BitVoxelMeaning voxel_meaning,
			bool* points_outside_map);

		template<class Voxel>
		__global__
		void kernelErode(Voxel* voxelmap_out, const Voxel* voxelmap_in, Vector3ui dimensions, float occupied_threshold, float erode_threshold);

		template<class Voxel>
		__global__
		void kernelInsertMetaPointCloud(Voxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			BitVoxelMeaning voxel_meaning, Vector3ui dimensions, float voxel_side_length,
			bool* points_outside_map);

		template<>
		__global__
		void kernelInsertMetaPointCloud(DistanceVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			BitVoxelMeaning voxel_meaning, Vector3ui dimensions, float voxel_side_length,
			bool* points_outside_map);

		template<class Voxel>
		__global__
		void kernelInsertMetaPointCloud(Voxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			BitVoxelMeaning* voxel_meanings, Vector3ui map_dim,
			float voxel_side_length,
			bool* points_outside_map);

		template<>
		__global__
		void kernelInsertMetaPointCloud(DistanceVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			BitVoxelMeaning* voxel_meanings, Vector3ui map_dim,
			float voxel_side_length,
			bool* points_outside_map);

		template<class BitVectorVoxel>
		__global__
		void kernelInsertMetaPointCloudSelfCollCheck(BitVectorVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			const BitVoxelMeaning* voxel_meanings, Vector3ui dimensions, unsigned int sub_cloud,
			float voxel_side_length, const BitVector<BIT_VECTOR_LENGTH>* coll_masks,
			bool* points_outside_map, BitVector<BIT_VECTOR_LENGTH>* colliding_subclouds);

		template<>
		__global__
		void kernelInsertMetaPointCloudSelfCollCheck(BitVectorVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
			const BitVoxelMeaning* voxel_meanings, Vector3ui dimensions, unsigned int sub_cloud,
			float voxel_side_length, const BitVector<BIT_VECTOR_LENGTH>* coll_masks,
			bool* points_outside_map, BitVector<BIT_VECTOR_LENGTH>* colliding_subclouds);

		/**
		 * Shifts all swept-volume-IDs by shift_size towards lower IDs.
		 * Currently this is limited to a shift size <64
		 */
		template<std::size_t length>
		__global__
		void kernelShiftBitVector(BitVoxel<length>* voxelmap, uint32_t voxelmap_size, uint8_t shift_size);

		/**
		 * cjuelg: jump flood distances, obstacle vectors
		 */
		__global__
		void kernelJumpFlood3D(const DistanceVoxel* __restrict__ voxels_input, DistanceVoxel* __restrict__ voxels_output, Vector3ui dims, int32_t step_width);

		/**
		 * cjuelg: brute force exact obstacle distances
		 */
		__global__
		void kernelExactDistances3D(DistanceVoxel* voxels, Vector3ui dims, float voxel_side_length,
				Vector3f* obstacles, std::size_t num_obstacles);


		struct OccupiedVoxels
		{
			__host__ __device__ uint32_t operator()(const BitVectorVoxel& val);
		};

		struct CullHiddenVoxels
		{
			const uint32_t* m_data;
			Vector3ui m_dim;

			__host__ __device__
				CullHiddenVoxels(const uint32_t* data, Vector3ui dims);

			__host__ __device__ uint32_t operator()(const uint32_t& idx);
		};

		struct GatherCompacted
		{
			Vector3ui m_dim;
			Vector3ui* m_data;

			__host__ __device__
				GatherCompacted(Vector3ui* data, Vector3ui dim);

			//tuple<filter mask, compaction mask, index>
			__host__ __device__ void operator()(const thrust::tuple<uint32_t, uint32_t, uint32_t>& input);
		};

		/*
		 * creates a mask of 1s and 0s based on the occupation of voxels in the vector
		 */
		thrust::device_vector<uint32_t> getOccupationVoxels(const thrust::device_vector<BitVectorVoxel>& dev_data);

		/*
		 * assumes a mask of 1s and 0s adding them up
		 */
		uint32_t getOccupied(const thrust::device_vector<uint32_t>& vec);

		/*
		 * same as getOccupied but with resulting compaction indices in input array as output
		 */
		uint32_t getOccupied_2(thrust::device_vector<uint32_t>& in_out);

		/*
		 * Removes 1s from a voxel filter if it is surrounded by other voxels
		 */
		thrust::device_vector<uint32_t> culled_filter(const thrust::device_vector<uint32_t>& filter, const Vector3ui& dim);

		/*
		 * returns all occupied Voxels coordinates which are not surrounded by other voxels
		 */
		template<class Voxel>
		class TemplateVoxelMap;

		template<class Voxel>
		std::vector<Vector3ui> extract_visual_voxels(const TemplateVoxelMap<Voxel>& in);

		/*
		 * returns all occupied Voxels coordinates which are not surrounded by other voxels
		 */
		std::vector<Vector3ui> extract_visual_voxels(const thrust::device_vector<BitVectorVoxel>& in, const Vector3ui& dim);
	} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif