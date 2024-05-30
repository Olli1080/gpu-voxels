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
 * \date    2014-07-10
 *
 */
 //----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_H_INCLUDED

#include <vector>

#include <cuda_runtime.h>

#include <gpu_voxels/helpers/ThrustForward.h>
#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/voxelmap/AbstractVoxelMap.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>
#include <gpu_voxels/voxel/DefaultCollider.h>

/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Datastructure and according operations
 */
namespace gpu_voxels {
	namespace voxelmap {

		template<class Voxel>
		class TemplateVoxelMap : public AbstractVoxelMap
		{
		public:
			/*! Create a voxelmap that holds dim.x * dim.y * dim.z voxels.
			 *  A voxel is treated as cube with side length voxel_side_length. */
			TemplateVoxelMap(const Vector3ui& dim, float voxel_side_length, MapType map_type);

			/*!
			 * This constructor does NOT create a new voxel map on the GPU.
			 * The new object will represent the voxel map specified in /p dev_data.
			 * Warning: Not all member variables will be set correctly for the map.
			 */
			TemplateVoxelMap(Voxel* dev_data, Vector3ui dim, float voxel_side_length, MapType map_type);

			//! Destructor
			~TemplateVoxelMap() override;

			/* ======== getter functions ======== */

			//! get pointer to data array on device
			Voxel* getDeviceDataPtr();

			const Voxel* getConstDeviceDataPtr() const;

			const ThrustDeviceVector<Voxel>& getDeviceData() const;

			void* getVoidDeviceDataPtr() override;

			const void* getConstVoidDeviceDataPtr() const override;

			//! get the number of voxels held in the voxelmap
			uint32_t getVoxelMapSize() const
			{
				return m_dim.x() * m_dim.y() * m_dim.z();
			}

			//! get the side length of the voxels.
			float getVoxelSideLength() const override
			{
				return m_voxel_side_length;
			}

			/* ======== VoxelMap operations  ======== */
			/*! as above, without locking mutex, which then must be done manually!
			 * This might be necessary for combination with other operations to ensure
			 * that the map did not change since it was cleared.
			 */
			//void clearVoxelMapRemoteLock(BitVoxelMeaning voxel_meaning);

			//! print data array to screen for debugging (low performance)
			virtual void printVoxelMapData();

			virtual void gatherVoxelsByIndex(thrust::device_ptr<unsigned int> dev_indices_begin, thrust::device_ptr<unsigned int> dev_indices_end, thrust::device_ptr<Voxel> dev_output_begin);

			/* --- collision check operations --- */
			/*! Test for collision with other VoxelMap
			 *  with given occupancy thresholds.
			 *  Returns true if there is any collision.
			 *
			 *  Assumes same dimensions and voxel_side_length
			 *  as local VoxelMap. See also getDimensions() function.
			 */
			template< class OtherVoxel, class Collider>
			bool collisionCheck(TemplateVoxelMap<OtherVoxel>* other, Collider collider);


			//  __host__
			//  bool collisionCheckAlternative(const uint8_t threshold, VoxelMap* other,
			//          const uint8_t other_threshold, uint32_t loop_size);

			template< class OtherVoxel, class Collider>
			uint32_t collisionCheckWithCounter(TemplateVoxelMap<OtherVoxel>* other, Collider collider = DefaultCollider());

			template< class OtherVoxel, class Collider>
			uint32_t collisionCheckWithCounterRelativeTransform(const TemplateVoxelMap<OtherVoxel>* other, Collider collider = DefaultCollider(), const Vector3i& offset = Vector3i::Zero());

			//  __host__
			//  bool collisionCheckBoundingBox(uint8_t threshold, VoxelMap* other, uint8_t other_threshold,
			//                        Vector3ui bounding_box_start, Vector3ui bounding_box_end);


			  // ------ BEGIN Global API functions ------
			void insertPointCloud(const std::vector<Vector3f>& point_cloud, BitVoxelMeaning voxelmeaning) override;

			void insertPointCloud(const PointCloud& pointcloud, BitVoxelMeaning voxel_meaning) override;

			void insertPointCloud(const ThrustDeviceVector<Vector3f>& d_points, BitVoxelMeaning voxel_meaning) override;


			void insertCoordinateList(const std::vector<Vector3ui>& coordinates, BitVoxelMeaning voxel_meaning) override;

			void insertCoordinateList(const ThrustDeviceVector<Vector3ui>& d_coordinates, BitVoxelMeaning voxel_meaning) override;

			/**
			 * @brief insertMetaPointCloud Inserts a MetaPointCloud into the map.
			 * @param meta_point_cloud The MetaPointCloud to insert
			 * @param voxel_meaning Voxel meaning of all voxels
			 */
			void insertMetaPointCloud(const MetaPointCloud& meta_point_cloud, BitVoxelMeaning voxel_meaning) override;

			/**
			 * @brief insertMetaPointCloud Inserts a MetaPointCloud into the map. Each pointcloud
			 * inside the MetaPointCloud will get it's own voxel meaning as given in the voxel_meanings
			 * parameter. The number of pointclouds in the MetaPointCloud and the size of voxel_meanings
			 * have to be identical.
			 * @param meta_point_cloud The MetaPointCloud to insert
			 * @param voxel_meanings Vector with voxel meanings
			 */
			void insertMetaPointCloud(const MetaPointCloud& meta_point_cloud, const std::vector<BitVoxelMeaning>& voxel_meanings) override;

			bool merge(GpuVoxelsMapSharedPtr other, const Vector3f& metric_offset, const BitVoxelMeaning* new_meaning = nullptr) override;
			bool merge(GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset = Vector3i::Zero(), const BitVoxelMeaning* new_meaning = nullptr) override;

			/**
			 * @brief insertDilatedCoordinateList Dilates the given coordinates and stores the result in this voxelmap
			 * WARNING: Can lead to unexpected behavior due to race-conditions, when multiple coordinates try to dilate towards the same
			 *          neighbor concurrently. This will be unproblematic for voxel types that only spread "occupied/ not occupied" information
			 *          to neighbors.
			 * @param coordinates the coordinates to be dilated and inserted, in host memory
			 * @param voxel_meaning the voxel_meaning that will be used for inserted voxels
			 */
			virtual void insertDilatedCoordinateList(const std::vector<Vector3ui>& coordinates, BitVoxelMeaning voxel_meaning);

			/**
			 * @brief insertDilatedCoordinateList Dilates the given coordinates and stores the result in this voxelmap
			 * WARNING: Can lead to unexpected behavior due to race-conditions, when multiple coordinates try to dilate towards the same
			 *          neighbor concurrently. This will be unproblematic for voxel types that only spread "occupied/ not occupied" information
			 *          to neighbors.
			 * @param d_coordinates the coordinates to be dilated and inserted, in device memory
			 * @param voxel_meaning the voxel_meaning that will be used for inserted voxels
			 */
			virtual void insertDilatedCoordinateList(const ThrustDeviceVector<Vector3ui> d_coordinates, BitVoxelMeaning insert_voxel_meaning);

			virtual void insertClosedCoordinateList(const ThrustDeviceVector<Vector3ui>& d_coordinates, BitVoxelMeaning insert_voxel_meaning, float erode_threshold, float occupied_threshold, TemplateVoxelMap<Voxel>& buffer);
			virtual void insertClosedCoordinateList(const std::vector<Vector3ui>& coordinates, BitVoxelMeaning insert_voxel_meaning, float erode_threshold, float occupied_threshold, TemplateVoxelMap<Voxel>& buffer);

			virtual void insertClosedCoordinateList(const ThrustDeviceVector<Vector3ui>& d_coordinates, BitVoxelMeaning insert_voxel_meaning, float erode_threshold, float occupied_threshold = 0);
			virtual void insertClosedCoordinateList(const std::vector<Vector3ui>& coordinates, BitVoxelMeaning insert_voxel_meaning, float erode_threshold, float occupied_threshold = 0);

			/**
			 * @brief erodeInto Erodes this voxelmap and stores the result in the given voxelmap
			 * @param dest the destination voxelmap
			 * @param erode_threshold the minimum ratio of neighbors that have to be occupied for a voxel to be kept. 0.0 erodes no voxels,
			 *                        1.0 erodes voxels that have at least one free neighbor, FLT_EPSILON erodes only voxels that have no
			 *                        occupied neighbors
			 * @param occupied_threshold the threshold passed to Voxel::isOccupied to determine if the voxel is occupied
			 */
			virtual void erodeInto(TemplateVoxelMap<Voxel>& dest, float erode_threshold, float occupied_threshold = 0) const;

			/**
			 * @brief erodeLonelyInto Erodes all voxels that have no neighbors and stores the result in the given voxelmap
			 * @param dest the destination voxelmap
			 * @param occupied_threshold the threshold passed to Voxel::isOccupied to determine if the voxel is occupied
			 */
			virtual void erodeLonelyInto(TemplateVoxelMap<Voxel>& dest, float occupied_threshold = 0) const;

			std::size_t getMemoryUsage() const override
			{
				return m_dim.x() * m_dim.y() * m_dim.z() * sizeof(Voxel);
			}

			void clearMap() override;
			//! set voxel occupancies for a specific voxelmeaning to zero

			virtual Vector3f getCenterOfMass() const;
			virtual Vector3f getCenterOfMass(Vector3ui lower_bound, Vector3ui upper_bound) const;

			bool writeToDisk(std::string path) override;

			bool readFromDisk(std::string path) override;

			Vector3ui getDimensions() const override;

			Vector3f getMetricDimensions() const override;

			virtual void clone(const TemplateVoxelMap<Voxel>& other);

			// ------ END Global API functions ------


		protected:

			/* ======== Variables with content on host ======== */
			const Vector3ui m_dim;
			const Vector3f m_limits;
			float m_voxel_side_length;
			//uint32_t m_voxelmap_size;
			//uint32_t m_num_points;
			uint32_t m_blocks;
			uint32_t m_threads;
			uint32_t m_alternative_blocks;
			uint32_t m_alternative_threads;

			//! size of array for collision check
			uint32_t m_result_array_size;

			//! result array for collision check
			thrust::host_vector<bool> m_collision_check_results;
			//! result array for collision check with counter
			thrust::host_vector<uint16_t> m_collision_check_results_counter;

			//! performance measurement start time
			cudaEvent_t m_start;
			//! performance measurement stop time
			cudaEvent_t m_stop;
			//! performance measurement elapsed time
			float m_elapsed_time;

			/* ======== Variables with content on device ======== */
			struct CUDA_impl;
			std::unique_ptr<CUDA_impl> cuda_impl;

			void erode(Voxel* d_dest_data, const Voxel* d_src_data, float erode_threshold, float occupied_threshold) const;

			void insertDilatedCoordinateList(Voxel* d_dest_data, const ThrustDeviceVector<Vector3ui>& d_src_coordinates, BitVoxelMeaning voxel_meaning);

			void insertClosedCoordinateList(const ThrustDeviceVector<Vector3ui>& d_coordinates, BitVoxelMeaning insert_voxel_meaning, float erode_threshold, float occupied_threshold, Voxel* d_buffer);
		};

	} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif