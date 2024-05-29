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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-04
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_H
#define GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_H

#include <gpu_voxels/voxellist/AbstractVoxelList.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>

#include <gpu_voxels/voxel/DefaultCollider.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "gpu_voxels/voxelmap/TemplateVoxelMap.h"


/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Datastructure and according operations
 */
namespace gpu_voxels {
	namespace voxellist {

		template<class Voxel, class VoxelIDType>
		class TemplateVoxelList : public AbstractVoxelList
		{
			typedef typename thrust::device_vector<VoxelIDType>::iterator  keyIterator;

			typedef typename thrust::device_vector<Vector3ui>::iterator  coordIterator;
			typedef typename thrust::device_vector<Voxel>::iterator  voxelIterator;

			typedef thrust::tuple<coordIterator, voxelIterator> valuesIteratorTuple;
			typedef thrust::zip_iterator<valuesIteratorTuple> zipValuesIterator;

			typedef thrust::tuple<keyIterator, voxelIterator> keyVoxelIteratorTuple;
			typedef thrust::zip_iterator<keyVoxelIteratorTuple> keyVoxelZipIterator;

		public:
			typedef thrust::tuple<keyIterator, coordIterator, voxelIterator> keyCoordVoxelIteratorTriple;
			typedef thrust::zip_iterator<keyCoordVoxelIteratorTriple> keyCoordVoxelZipIterator;

			TemplateVoxelList(Vector3ui ref_map_dim, float voxel_side_length, MapType map_type);

			//! Destructor
			~TemplateVoxelList() override = default;

			/* ======== getter functions ======== */

			//! get thrust triple to the beggining of all data vectors
			virtual  keyCoordVoxelZipIterator getBeginTripleZipIterator();

			//! get thrust triple to the end of all data vectors
			virtual keyCoordVoxelZipIterator getEndTripleZipIterator();

			//! get access to data vectors on device
			typename thrust::device_vector<Voxel>::iterator getDeviceDataVectorBeginning()
			{
				return m_dev_list.begin();
			}
			typename thrust::device_vector<VoxelIDType>::iterator getDeviceIdVectorBeginning()
			{
				return m_dev_id_list.begin();
			}
			typename thrust::device_vector<Vector3ui>::iterator getDeviceCoordVectorBeginning()
			{
				return m_dev_coord_list.begin();
			}

			//! get pointer to data array on device
			Voxel* getDeviceDataPtr()
			{
				return thrust::raw_pointer_cast(m_dev_list.data());
			}
			const Voxel* getConstDeviceDataPtr() const
			{
				return thrust::raw_pointer_cast(m_dev_list.data());
			}
			VoxelIDType* getDeviceIdPtr()
			{
				return thrust::raw_pointer_cast(m_dev_id_list.data());
			}
			const VoxelIDType* getConstDeviceIdPtr() const
			{
				return thrust::raw_pointer_cast(m_dev_id_list.data());
			}
			thrust::device_vector<Vector3ui>& getDeviceCoords()
			{
				return m_dev_coord_list;
			}
			const thrust::device_vector<Vector3ui>& getDeviceCoords() const
			{
				return m_dev_coord_list;
			}
			Vector3ui* getDeviceCoordPtr()
			{
				return thrust::raw_pointer_cast(m_dev_coord_list.data());
			}
			const Vector3ui* getConstDeviceCoordPtr() const
			{
				return thrust::raw_pointer_cast(m_dev_coord_list.data());
			}

			void* getVoidDeviceDataPtr() override
			{
				return (void*)thrust::raw_pointer_cast(m_dev_list.data());
			}

			//! get the side length of the voxels.
			float getVoxelSideLength() const override
			{
				return m_voxel_side_length;
			}

			virtual void copyCoordsToHost(thrust::host_vector<Vector3ui>& host_vec);

			// ------ BEGIN Global API functions ------
			void insertPointCloud(const std::vector<Vector3f>& points, BitVoxelMeaning voxel_meaning) override;

			void insertPointCloud(const PointCloud& pointcloud, BitVoxelMeaning voxel_meaning) override;

			void insertPointCloud(const thrust::device_vector<Vector3f>& d_points, BitVoxelMeaning voxel_meaning) override;


			void insertCoordinateList(const std::vector<Vector3ui>& coordinates, BitVoxelMeaning voxel_meaning) override;

			void insertCoordinateList(const thrust::device_vector<Vector3ui>& d_coordinates, BitVoxelMeaning voxel_meaning) override;

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

			//TODO::
			bool insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud* robot_links,
			                                                const std::vector<BitVoxelMeaning>& voxel_meanings = {},
			                                                const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks = {},
			                                                BitVector<BIT_VECTOR_LENGTH>* colliding_meanings = nullptr) override;

			bool merge(const GpuVoxelsMapSharedPtr other, const Vector3f& metric_offset, const BitVoxelMeaning* new_meaning = nullptr) override;
			bool merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset = Vector3i::Zero(), const BitVoxelMeaning* new_meaning = nullptr) override;


			virtual bool subtract(const TemplateVoxelList<Voxel, VoxelIDType>* other, const Vector3f& metric_offset = Vector3f::Zero());
			virtual bool subtract(const TemplateVoxelList<Voxel, VoxelIDType>* other, const Vector3i& voxel_offset = Vector3i::Zero());

			virtual bool subtractFromCountingVoxelList(const TemplateVoxelList<BitVectorVoxel, VoxelIDType>* other, const Vector3f& metric_offset = Vector3f::Zero());
			virtual bool subtractFromCountingVoxelList(const TemplateVoxelList<BitVectorVoxel, VoxelIDType>* other, const Vector3i& voxel_offset = Vector3i::Zero());

			virtual void resize(size_t new_size);

			virtual void shrinkToFit();

			std::size_t getMemoryUsage() const override;

			void clearMap() override;
			//! set voxel occupancies for a specific voxelmeaning to zero

			virtual Vector3f getCenterOfMass() const;
			virtual Vector3f getCenterOfMass(Vector3ui lower_bound, Vector3ui upper_bound) const;

			bool writeToDisk(std::string path) override;

			bool readFromDisk(std::string path) override;

			Vector3ui getDimensions() const override;

			Vector3f getMetricDimensions() const override;

			// ------ END Global API functions ------

			/**
			 * @brief extractCubes Extracts a cube list for visualization
			 * @param [out] output_vector Resulting cube list
			 */
			virtual void extractCubes(thrust::device_vector<Cube>** output_vector) const;

			/**
			 * @brief collideVoxellists Internal binary search between voxellists
			 * @param other Other Voxellist
			 * @param offset Offset of other map to this map
			 * @param collision_stencil Binary vector storing the collisions. Has to be the size of 'this'
			 * @return Number of collisions
			 */
			 // virtual size_t collideVoxellists(const TemplateVoxelList<Voxel, VoxelIDType> *other, const Vector3i &offset,
			 //                                  thrust::device_vector<bool>& collision_stencil) const;
			
			size_t collideVoxellists(const TemplateVoxelList<ProbabilisticVoxel, VoxelIDType>* other, const Vector3i& offset,
				thrust::device_vector<bool>& collision_stencil) const;

			template<size_t length>
			size_t collideVoxellists(const TemplateVoxelList<BitVoxel<length>, VoxelIDType>* other, const Vector3i& offset,
				thrust::device_vector<bool>& collision_stencil) const;
			
			size_t collideVoxellists(const TemplateVoxelList<CountingVoxel, VoxelIDType>* other, const Vector3i& offset,
				thrust::device_vector<bool>& collision_stencil) const;

			/**
			 * @brief collisionCheckWithCollider
			 * @param other Other VoxelList
			 * @param collider Collider object
			 * @param offset Offset of other map to this map
			 * @return number of collisions
			 */
			//template<class OtherVoxel, class Collider>
			//size_t collisionCheckWithCollider(const TemplateVoxelList<OtherVoxel, VoxelIDType>* other, Collider collider = DefaultCollider(), const Vector3i& offset = Vector3i::Zero());

			/**
			 * @brief collisionCheckWithCollider
			 * @param other Other voxelmap
			 * @param collider Collider object
			 * @param offset Offset of other map to this map
			 * @return number of collisions
			 */
			template< class OtherVoxel, class Collider>
			size_t collisionCheckWithCollider(const voxelmap::TemplateVoxelMap<OtherVoxel>* other, Collider collider = DefaultCollider(), const Vector3i& offset = Vector3i::Zero());

			/**
			 * @brief equals compares two voxellists by their elements.
			 * @return true, if all elements are equal, false otherwise
			 */
			template< class OtherVoxel, class OtherVoxelIDType>
			bool equals(const TemplateVoxelList<OtherVoxel, OtherVoxelIDType>& other) const;

			/**
			 * @brief screendump Prints ALL elemets of the list to the screen
			 */
			virtual void screendump(bool with_voxel_content = true) const;

			virtual void clone(const TemplateVoxelList<Voxel, VoxelIDType>& other);

			struct VoxelToCube
			{
				VoxelToCube() = default;

				template<size_t length>
				__host__ __device__
				Cube operator()(const Vector3ui& coords, const BitVoxel<length>& voxel) const {

					return { 1, coords, voxel.bitVector() };
				}
				__host__ __device__
				Cube operator()(const Vector3ui& coords, const CountingVoxel& voxel) const {

					if (voxel.getCount() > 0)
					{
						return { 1, coords, eBVM_OCCUPIED };
					}
					else
					{
						return { 1, coords, eBVM_FREE };
					}
				}
				__host__ __device__
				Cube operator()(const Vector3ui& coords, const ProbabilisticVoxel& voxel) const {
					return { 1, coords, static_cast<BitVoxelMeaning>(voxel.getOccupancy() * eBVM_MAX_OCC_PROB) };
				}
			};

			/* ======== Variables with content on device ======== */
			/* Follow the Thrust paradigm: Struct of Vectors */
			/* need to be public in order to be accessed by TemplateVoxelLists with other template arguments*/
			thrust::device_vector<VoxelIDType> m_dev_id_list;  // contains the voxel addresses / morton codes (This can not be a Voxel*, as Thrust can not sort pointers)
			thrust::device_vector<Vector3ui> m_dev_coord_list; // contains the voxel metric coordinates
			thrust::device_vector<Voxel> m_dev_list;           // contains the actual data: bitvector or probability

		protected:

			virtual void remove_out_of_bounds();
			virtual void make_unique();

			/* ======== Variables with content on host ======== */
			float m_voxel_side_length;
			Vector3ui m_ref_map_dim;

			uint32_t m_blocks;
			uint32_t m_threads;

			//! result array for collision check
			thrust::host_vector<bool> m_collision_check_results;
			//! result array for collision check with counter
			thrust::host_vector<uint16_t> m_collision_check_results_counter;


			//! results of collision check on device
			thrust::device_vector<bool> m_dev_collision_check_results;

			//! result array for collision check with counter on device
			thrust::device_vector<uint16_t> m_dev_collision_check_results_counter;
		};

		extern template bool TemplateVoxelList<CountingVoxel, MapVoxelID>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning);
		extern template bool TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning);
	} // end of namespace voxellist
} // end of namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_H