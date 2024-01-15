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

#ifndef GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_HPP
#define GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_HPP

#include "TemplateVoxelList.h"

#include <fstream>
#include <gpu_voxels/logging/logging_voxellist.h>
#include <gpu_voxels/voxellist/kernels/VoxelListOperations.hpp>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>
#include <gpu_voxels/voxelmap/BitVoxelMap.hpp>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include <thrust/system_error.h>

namespace gpu_voxels
{
	namespace voxellist
	{

		// Thrust Operator Merge:
		template<class Voxel, class VoxelIDType>
		struct Merge : public thrust::binary_function<thrust::tuple<VoxelIDType, Voxel>,
			thrust::tuple<VoxelIDType, Voxel>, thrust::tuple<VoxelIDType, Voxel> >
		{
			typedef thrust::tuple<VoxelIDType, Voxel> keyVoxelTuple;

			__host__ __device__
				keyVoxelTuple operator()(const keyVoxelTuple& lhs, const keyVoxelTuple& rhs) const
			{
				VoxelIDType l_key = thrust::get<0>(lhs);
				Voxel l_voxel = thrust::get<1>(lhs);

				VoxelIDType r_key = thrust::get<0>(rhs);
				Voxel r_voxel = thrust::get<1>(rhs);

				keyVoxelTuple ret = rhs;

				if (l_key == r_key)
					thrust::get<1>(ret) = Voxel::reduce(l_voxel, r_voxel);

				return ret;
			}
		};

		// Thrust operator applyOffsetOperator:
		template<class VoxelIDType>
		struct applyOffsetOperator : public thrust::unary_function<thrust::tuple<Vector3ui, VoxelIDType>,
			thrust::tuple<Vector3ui, VoxelIDType>>
		{
			typedef thrust::tuple<Vector3ui, VoxelIDType> coordKeyTuple;

			ptrdiff_t addr_offset;
			Vector3i coord_offset;
			applyOffsetOperator(const Vector3ui& ref_map_dim, const Vector3i& offset)
			{
				coord_offset = offset;
				addr_offset = voxelmap::getVoxelIndexSigned(ref_map_dim, offset);
			}

			__host__ __device__
				coordKeyTuple operator()(const coordKeyTuple& input) const
			{
				coordKeyTuple ret;
				thrust::get<0>(ret) = (thrust::get<0>(input).template cast<int64_t>() + coord_offset.template cast<int64_t>()).template cast<uint32_t>();
				thrust::get<1>(ret) = thrust::get<1>(input) + addr_offset;
				return ret;
			}
		};

		template<class Voxel, class VoxelIDType>
		TemplateVoxelList<Voxel, VoxelIDType>::TemplateVoxelList(const Vector3ui ref_map_dim, const float voxel_side_length, const MapType map_type)
			: m_voxel_side_length(voxel_side_length),
			m_ref_map_dim(ref_map_dim),
			m_collision_check_results(cMAX_NR_OF_BLOCKS, false),
			m_collision_check_results_counter(cMAX_NR_OF_BLOCKS, 0)
		{
			this->m_map_type = map_type;

			// copy initialized arrays to device
			m_dev_collision_check_results = m_collision_check_results;
			m_dev_collision_check_results_counter = m_collision_check_results_counter;
		}

		/*!
		 * \brief TemplateVoxelList<Voxel>::make_unique
		 * Sorting stuff and make unique
		 * This has to be executed in that order, as "unique" only removes duplicates if they appear in a row!!
		 * The keys and the values are sorted according to the key.
		 * After sorting, the bitvectors of successive voxels with the same key are merged.
		 * After that the keys are unified. According values are dropped too.
		 */
		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::make_unique()
		{
			// Sort all entries by key.
			try
			{
				LOGGING_DEBUG_C(VoxellistLog, TemplateVoxelList, "List size before make_unique: " << m_dev_list.size() << endl);


				//    thrust::host_vector<VoxelIDType> dev_id_list_h(m_dev_id_list);
				//    thrust::host_vector<Vector3ui> dev_coord_list_h(m_dev_coord_list);


				//    for(uint i = 0; i < dev_id_list_h.size(); i++)
				//    {
				//        if(i%10 == 0)
				//        {
				//            std::cout << "ID: " << dev_id_list_h[i] << std::endl <<
				//                         "Coord List: [" << dev_coord_list_h[i].x << ", " << dev_coord_list_h[i].y << ", " << dev_coord_list_h[i].z << "]" << std::endl;
				//        }
				//    }


					// the ZipIterator represents the data that is sorted by the keys in m_dev_id_list
				thrust::sort_by_key(m_dev_id_list.begin(), m_dev_id_list.end(),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin(), m_dev_list.begin())),
					thrust::less<VoxelIDType>());
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			}
			catch (thrust::system_error& e)
			{
				LOGGING_ERROR(VoxellistLog, "GetLastError: " << cudaGetLastError() << endl);
				//    cudaDeviceReset();
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception while sorting: " << e.what() << endl);
				exit(-1);
			}
			try
			{
				// Reverse iterate over sorted entries and merge successive voxel-bitvectors into the predecessor
				// of voxels with the same key. We dont touch the coordinates as they are the same either.
				thrust::inclusive_scan(thrust::make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(m_dev_id_list.end(), m_dev_list.end()))),
					thrust::make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(m_dev_id_list.begin(), m_dev_list.begin()))),
					thrust::make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(m_dev_id_list.end(), m_dev_list.end()))),
					Merge<Voxel, VoxelIDType>());
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			}
			catch (thrust::system_error& e)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception while doing inclusive_scan: " << e.what() << endl);
				exit(-1);
			}
			// Now drop all duplicates.
			// This will remove successors and keep the first entry with the merged bitvectors.
			try
			{
				 thrust::pair<keyIterator, zipValuesIterator> new_end = thrust::unique_by_key(m_dev_id_list.begin(), m_dev_id_list.end(),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin(), m_dev_list.begin())));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				const size_t new_length = thrust::distance(m_dev_id_list.begin(), new_end.first);
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				this->resize(new_length);
			}
			catch (thrust::system_error& e)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception while dropping duplicates: " << e.what() << endl);
				exit(-1);
			}
			LOGGING_DEBUG_C(VoxellistLog, TemplateVoxelList, "List size after make_unique: " << m_dev_list.size() << endl);
		}
		
		template<class Voxel, class VoxelIDType>
		size_t TemplateVoxelList<Voxel, VoxelIDType>::collideVoxellists(const TemplateVoxelList<ProbabilisticVoxel, VoxelIDType>* other,
			const Vector3i& offset, thrust::device_vector<bool>& collision_stencil) const
		{
			LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "colliding any VoxelList with a ProbabilisticVoxelList is no supported! Not performing collision check." << endl);
			return 0;
		}

		template<class Voxel, class VoxelIDType>
		size_t TemplateVoxelList<Voxel, VoxelIDType>::collideVoxellists(const TemplateVoxelList<CountingVoxel, VoxelIDType>* other,
			const Vector3i& offset, thrust::device_vector<bool>& collision_stencil) const
		{
			LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "colliding any VoxelList with a CountingVoxelList is no supported! Not performing collision check." << endl);
			return 0;
		}

		struct logicalAND {

			__host__ __device__
				bool operator()(bool a, bool b)
			{
				return a && b;
			}
		};

		template<class Voxel, class VoxelIDType>
		size_t TemplateVoxelList<Voxel, VoxelIDType>::collideVoxellists(const TemplateVoxelList<BitVectorVoxel, VoxelIDType>* other,
			const Vector3i& offset, thrust::device_vector<bool>& collision_stencil) const
		{
			std::scoped_lock lock(this->m_mutex, other->m_mutex);

			// searching for the elements of "other" in "this". Therefore stencil has to be the size of "this"
			try
			{
				//will contain a stencil of elements that should be considered for collision checking
				//TODO: initialise to empty if not CVL
				thrust::device_vector<bool> filtermask_device(collision_stencil.size());

				// if offset is given, we need our own comparison operator which is a lot slower than the comparison on built-in data types!
				// See: http://stackoverflow.com/questions/9037906/fast-cuda-thrust-custom-comparison-operator
				if (offset != Vector3i::Zero())
				{
					LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Offset for VoxelList collision not supported! Not performing collision check." << endl);
					return 0;
				}
				else 
				{
					//only the counting voxellist implements the collision_stencil as a filter mask
					if (this->m_map_type == MT_COUNTING_VOXELLIST)
						//if the collision_stencil contains a true value it will be treated as a filter mask
					{
						thrust::copy(collision_stencil.begin(), collision_stencil.end(), filtermask_device.begin());
						HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
					}

					thrust::binary_search(thrust::device,
						other->m_dev_id_list.begin(), other->m_dev_id_list.end(),
						m_dev_id_list.begin(), m_dev_id_list.end(),
						collision_stencil.begin());
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				if (this->m_map_type == MT_COUNTING_VOXELLIST)
				{ // is used to count the collisions, without overriding the collision_stencil, because it is used in the subtract methods
					thrust::device_vector<bool> count_device(collision_stencil.size());

					//TODO: why does this not zero out all collisions in case of CVL.subtractFromCVL(BVL)?
					thrust::transform(collision_stencil.begin(), collision_stencil.end(), filtermask_device.begin(), count_device.begin(), logicalAND());
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

					return thrust::count(count_device.begin(), count_device.end(), true);
				}
				else
				{
					return thrust::count(collision_stencil.begin(), collision_stencil.end(), true);
				}
			}
			catch (thrust::system_error& e)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception " << e.what() << endl);
				exit(-1);
			}
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::copyCoordsToHost(thrust::host_vector<Vector3ui>& host_vec)
		{
			host_vec = m_dev_coord_list;
		}

		template<class Voxel, class VoxelIDType>
		struct is_out_of_bounds
		{
			Vector3ui dims;
			is_out_of_bounds(Vector3ui map_dims) : dims(map_dims) {}

			__host__ __device__
				bool operator()(thrust::tuple<VoxelIDType, Vector3ui, Voxel> triple_it) const
			{
				const Vector3ui v = thrust::get<1>(triple_it);
				//printf("voxel count was: %d, id: %d\n", v, thrust::get<0>(triple_it));

				return (v.x() >= dims.x()) || (v.y() >= dims.y()) || (v.z() >= dims.z());
			}
		};

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::remove_out_of_bounds()
		{
			//this->screendump(true); // DEBUG

			std::lock_guard guard(this->m_mutex);

			// find the overlapping voxels
			keyCoordVoxelZipIterator new_end;

			is_out_of_bounds<Voxel, VoxelIDType> filter(m_ref_map_dim);

			// remove voxels below threshold
			new_end = thrust::remove_if(this->getBeginTripleZipIterator(),
				this->getEndTripleZipIterator(),
				filter);

			const size_t new_length = thrust::distance(m_dev_id_list.begin(), thrust::get<0>(new_end.get_iterator_tuple()));
			this->resize(new_length);
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			//this->screendump(true); // DEBUG
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::insertPointCloud(const std::vector<Vector3f>& points, const BitVoxelMeaning voxel_meaning)
		{
			const thrust::device_vector<Vector3f> d_points = { points.begin(), points.end() };
			insertPointCloud(d_points, voxel_meaning);
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::insertPointCloud(const PointCloud& pointcloud, const BitVoxelMeaning voxel_meaning)
		{
			insertPointCloud(pointcloud.getPointsDevice(), voxel_meaning);
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::insertPointCloud(const thrust::device_vector<Vector3f>& d_points, const BitVoxelMeaning voxel_meaning)
		{
			if (d_points.empty())
				return;

			std::lock_guard guard(this->m_mutex);

			auto offset_new_entries = static_cast<uint32_t>(m_dev_list.size());

			// resize capacity
			this->resize(offset_new_entries + d_points.size());

			// get raw pointers to the thrust vectors data:
			Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
			Vector3ui* dev_coord_list_ptr = thrust::raw_pointer_cast(m_dev_coord_list.data());
			VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

			// copy points to the gpu
			uint32_t num_blocks, threads_per_block;
			computeLinearLoad(d_points.size(), num_blocks, threads_per_block);
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			kernelInsertGlobalPointCloud<<<num_blocks, threads_per_block>>>(dev_id_list_ptr, dev_coord_list_ptr, dev_voxel_list_ptr,
				m_ref_map_dim, m_voxel_side_length,
				d_points.data().get(), d_points.size(), offset_new_entries, voxel_meaning);
			CHECK_CUDA_ERROR();
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			remove_out_of_bounds();

			make_unique();
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::insertCoordinateList(const std::vector<Vector3ui>& coordinates, const BitVoxelMeaning voxel_meaning)
		{
			const thrust::device_vector<Vector3ui> d_coordinates = { coordinates.begin(), coordinates.end() };
			insertCoordinateList(d_coordinates, voxel_meaning);
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::insertCoordinateList(const thrust::device_vector<Vector3ui>& d_coordinates, const BitVoxelMeaning voxel_meaning)
		{
			if (d_coordinates.empty())
				return;

			std::lock_guard guard(this->m_mutex);

			uint32_t offset_new_entries = static_cast<uint32_t>(m_dev_list.size());

			// resize capacity
			this->resize(offset_new_entries + d_coordinates.size());

			// get raw pointers to the thrust vectors data:
			Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
			Vector3ui* dev_coord_list_ptr = thrust::raw_pointer_cast(m_dev_coord_list.data());
			VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

			// copy points to the gpu
			uint32_t num_blocks, threads_per_block;
			computeLinearLoad(d_coordinates.size(), num_blocks, threads_per_block);
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			kernelInsertCoordinateTuples<<<num_blocks, threads_per_block>>>(dev_id_list_ptr, dev_coord_list_ptr, dev_voxel_list_ptr,
				m_ref_map_dim, d_coordinates.data().get(), d_coordinates.size(), offset_new_entries, voxel_meaning);
			CHECK_CUDA_ERROR();
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			remove_out_of_bounds();

			make_unique();
		}


		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::insertMetaPointCloud(const MetaPointCloud& meta_point_cloud, BitVoxelMeaning voxel_meaning)
		{
			std::lock_guard guard(this->m_mutex);

			const uint32_t total_points = meta_point_cloud.getAccumulatedPointcloudSize();

			if (total_points <= 0)
				return;

			uint32_t offset_new_entries = m_dev_list.size();
			// resize capacity
			this->resize(offset_new_entries + total_points);

			// get raw pointers to the thrust vectors data:
			Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
			Vector3ui* dev_coord_list_ptr = thrust::raw_pointer_cast(m_dev_coord_list.data());
			VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

			computeLinearLoad(total_points, m_blocks, m_threads);
			kernelInsertMetaPointCloud<<<m_blocks, m_threads>>>(dev_id_list_ptr, dev_coord_list_ptr, dev_voxel_list_ptr,
				m_ref_map_dim, m_voxel_side_length,
				meta_point_cloud.getDeviceConstPointer().get(),
				offset_new_entries, voxel_meaning);
			CHECK_CUDA_ERROR();
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			remove_out_of_bounds();

			make_unique();
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::insertMetaPointCloud(const MetaPointCloud& meta_point_cloud,
			const std::vector<BitVoxelMeaning>& voxel_meanings)
		{
			if (meta_point_cloud.getNumberOfPointclouds() != voxel_meanings.size())
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Number of VoxelMeanings differs from number of Sub-Pointclouds! Not inserting MetaPointCloud!" << endl);
				return;
			}

			std::lock_guard guard(this->m_mutex);
			const uint32_t total_points = meta_point_cloud.getAccumulatedPointcloudSize();

			uint32_t offset_new_entries = m_dev_list.size();
			// resize capacity
			this->resize(offset_new_entries + total_points);

			// get raw pointers to the thrust vectors data:
			Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
			Vector3ui* dev_coord_list_ptr = thrust::raw_pointer_cast(m_dev_coord_list.data());
			VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

			BitVoxelMeaning* dev_voxel_meanings;
			const size_t size = voxel_meanings.size() * sizeof(BitVoxelMeaning);
			HANDLE_CUDA_ERROR(cudaMalloc(&dev_voxel_meanings, size));
			HANDLE_CUDA_ERROR(cudaMemcpy(dev_voxel_meanings, voxel_meanings.data(), size, cudaMemcpyHostToDevice));

			computeLinearLoad(total_points, m_blocks, m_threads);
			kernelInsertMetaPointCloud<<<m_blocks, m_threads>>>(dev_id_list_ptr, dev_coord_list_ptr, dev_voxel_list_ptr,
				m_ref_map_dim, m_voxel_side_length,
				meta_point_cloud.getDeviceConstPointer().get(),
				offset_new_entries, dev_voxel_meanings);
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			HANDLE_CUDA_ERROR(cudaFree(dev_voxel_meanings));

			remove_out_of_bounds();

			make_unique();
		}

		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud* robot_links,
			const std::vector<BitVoxelMeaning>& voxel_meanings,
			const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks,
			BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
		{
			LOGGING_ERROR_C(VoxellistLog, CountingVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
			return true;
		}


		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::resize(size_t new_size)
		{
			std::lock_guard guard(this->m_mutex);
			m_dev_list.resize(new_size);
			m_dev_coord_list.resize(new_size);
			m_dev_id_list.resize(new_size);
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::shrinkToFit()
		{
			std::lock_guard guard(this->m_mutex);
			m_dev_list.shrink_to_fit();
			m_dev_coord_list.shrink_to_fit();
			m_dev_id_list.shrink_to_fit();
		}

		template<class Voxel, class VoxelIDType>
		size_t TemplateVoxelList<Voxel, VoxelIDType>::getMemoryUsage() const
		{
			std::lock_guard guard(this->m_mutex);
			const size_t ret = m_dev_list.size() * sizeof(Voxel) +
				m_dev_coord_list.size() * sizeof(Vector3ui) +
				m_dev_id_list.size() * sizeof(VoxelIDType);
			return ret;
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::clearMap()
		{
			std::lock_guard guard(this->m_mutex);
			m_dev_list.clear();
			m_dev_coord_list.clear();
			m_dev_id_list.clear();
		}

		struct filterByBoundaries
		{
			Vector3ui upper_bound;
			Vector3ui lower_bound;

			filterByBoundaries(Vector3ui upper, Vector3ui lower) : upper_bound(upper), lower_bound(lower) {}

			__host__ __device__
				bool operator()(Vector3ui a)
			{
				const bool inBound = a.x() >= lower_bound.x() && a.y() >= lower_bound.y() && a.z() >= lower_bound.z()
					&& a.x() < upper_bound.x() && a.y() < upper_bound.y() && a.z() < upper_bound.z();
				return inBound;
			}
		};

		struct sumVector3ui
		{
			__host__ __device__
			Vector3ui operator()(Vector3ui a, Vector3ui b)
			{
				return {a.x() + b.x(), a.y() + b.y(), a.z() + b.z() };
			}
		};

		template<class Voxel, class VoxelIDType>
		Vector3f TemplateVoxelList<Voxel, VoxelIDType>::getCenterOfMass() const
		{
			const Vector3ui lower(0, 0, 0);
			const Vector3ui upper = m_ref_map_dim;
			return getCenterOfMass(lower, upper);
		}

		template<class Voxel, class VoxelIDType>
		Vector3f TemplateVoxelList<Voxel, VoxelIDType>::getCenterOfMass(Vector3ui lower_bound, Vector3ui upper_bound) const
		{
			const filterByBoundaries filter(upper_bound, lower_bound);
			const int voxelCount = thrust::count_if(m_dev_coord_list.begin(), m_dev_coord_list.end(), filter);

			if (voxelCount <= 0)
			{
				LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "No Voxels Found in given Bounding Box: Lower Bound ("
					<< lower_bound.x() << "|" << lower_bound.y() << "|" << lower_bound.z() << ")  "
					<< "Upper Bound (" << upper_bound.x() << "|" << upper_bound.y() << "|" << upper_bound.z() << ")" << endl);

				return {};
			}

			//filter by axis alligned bounding box
			thrust::device_vector<Vector3ui> inBoundaries(voxelCount);
			thrust::copy_if(m_dev_coord_list.begin(), m_dev_coord_list.end(), inBoundaries.begin(), filter);

			//calculate sum of all voxelpositions
			Vector3ui tmp = Vector3ui::Zero();
			const Vector3ui sumVector = thrust::reduce(inBoundaries.begin(), inBoundaries.end(), tmp, sumVector3ui());
			//divide by voxel count
			const Vector3f metricSum = sumVector.template cast<float>() * m_voxel_side_length;
			const Vector3f coM = Vector3f(metricSum.x() / voxelCount, metricSum.y() / voxelCount, metricSum.z() / voxelCount);
			return coM;
		}

		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::writeToDisk(const std::string path)
		{

			LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "Dumping VoxelList to disk: " <<
				getDimensions().x() << " Voxels ==> " << (getMemoryUsage() * cBYTE2MBYTE) << " MB. ..." << endl);

			std::lock_guard guard(this->m_mutex);
			std::ofstream out(path.c_str());

			if (!out.is_open())
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Write to file " << path << " failed!" << endl);
				return false;
			}
			thrust::host_vector<VoxelIDType> host_id_list = m_dev_id_list;
			thrust::host_vector<Vector3ui> host_coord_list = m_dev_coord_list;
			thrust::host_vector<Voxel> host_list = m_dev_list;

			uint32_t num_voxels = host_list.size();
			int32_t map_type = m_map_type;

			out.write((char*)&map_type, sizeof(int32_t));
			out.write((char*)&m_ref_map_dim, sizeof(Vector3ui));
			out.write((char*)&m_voxel_side_length, sizeof(float));
			out.write((char*)&num_voxels, sizeof(uint32_t));
			out.write((char*)host_id_list.data(), num_voxels * sizeof(VoxelIDType));
			out.write((char*)host_coord_list.data(), num_voxels * sizeof(Vector3ui));
			out.write((char*)host_list.data(), num_voxels * sizeof(Voxel));

			out.close();
			LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "Write to disk done: Extracted " << num_voxels << " Voxels." << endl);
			return true;
		}

		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::readFromDisk(const std::string path)
		{
			std::lock_guard guard(this->m_mutex);
			thrust::host_vector<VoxelIDType> host_id_list;
			thrust::host_vector<Vector3ui> host_coord_list;
			thrust::host_vector<Voxel> host_list;

			uint32_t num_voxels;
			float voxel_side_length;
			Vector3ui ref_map_dim;
			int32_t map_type;

			std::ifstream in(path.c_str());
			if (!in.is_open())
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Read from file " << path << " failed!" << endl);
				return false;
			}

			in.read((char*)&map_type, sizeof(int32_t));
			if (map_type != m_map_type)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Read from file failed: The map type (" << map_type << ") does not match current object (" << m_map_type << ")!" << endl);
				return false;
			}
			in.read((char*)&ref_map_dim, sizeof(Vector3ui));
			if (ref_map_dim != m_ref_map_dim)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Read from file failed: Read reference map dimension (" << ref_map_dim << ") does not match current object (" << m_ref_map_dim << ")!" << endl);
				return false;
			}
			in.read((char*)&voxel_side_length, sizeof(float));
			if (voxel_side_length != m_voxel_side_length)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Read from file failed: Read Voxel side length (" << voxel_side_length << ") does not match current object (" << m_voxel_side_length << ")!" << endl);
				return false;
			}
			in.read((char*)&num_voxels, sizeof(uint32_t));

			host_id_list.resize(num_voxels);
			host_coord_list.resize(num_voxels);
			host_list.resize(num_voxels);
			in.read((char*)host_id_list.data(), num_voxels * sizeof(VoxelIDType));
			in.read((char*)host_coord_list.data(), num_voxels * sizeof(Vector3ui));
			in.read((char*)host_list.data(), num_voxels * sizeof(Voxel));

			in.close();
			LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "Read " << num_voxels << " Voxels from file." << endl;);


			m_dev_id_list = host_id_list;
			m_dev_coord_list = host_coord_list;
			m_dev_list = host_list;
			return true;
		}

		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning)
		{
			LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
			return false;
		}

		template<>
		bool TemplateVoxelList<CountingVoxel, MapVoxelID>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning)
		{
			switch (other->getMapType())
			{
			case MT_BITVECTOR_VOXELLIST:
			{
				auto* m = other->as<BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>>();
				std::scoped_lock lock(this->m_mutex, m->m_mutex);

				const uint32_t num_new_voxels = m->getDimensions().x();
				const uint32_t offset_new_entries = m_dev_list.size();
				// resize capacity
				this->resize(offset_new_entries + num_new_voxels);

				// We append the given list to our own list of points.
				thrust::copy(
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_coord_list.begin(), m->m_dev_id_list.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_coord_list.end(), m->m_dev_id_list.end())),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				make_unique();

				return true;
			}

			case MT_COUNTING_VOXELLIST:
			{
				auto* m = other->as<CountingVoxelList>();
				std::scoped_lock lock(this->m_mutex, m->m_mutex);

				const uint32_t num_new_voxels = m->getDimensions().x();
				const uint32_t offset_new_entries = m_dev_list.size();
				// resize capacity
				this->resize(offset_new_entries + num_new_voxels);

				// We append the given list to our own list of points.
				thrust::copy(
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_list.begin(), m->m_dev_coord_list.begin(), m->m_dev_id_list.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_list.end(), m->m_dev_coord_list.end(), m->m_dev_id_list.end())),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_list.begin() + offset_new_entries, m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				make_unique();

				return true;
			}
			default:
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
				return false;
			}
			}
		}

		template<typename Voxel>
		struct is_occupied
		{
			is_occupied(float occupied_threshold)
				: occupied_threshold(occupied_threshold)
			{
			}

			__host__ __device__
				bool operator()(const Voxel& voxel)
			{
				return voxel.isOccupied(occupied_threshold);
			}

			float occupied_threshold;
		};

		template<typename Voxel>
		struct voxelid_to_voxelcoord
		{
			voxelid_to_voxelcoord(const Voxel* base_ptr, Vector3ui dim)
				: base_ptr(base_ptr)
				, dim(dim)
			{
			}

			__host__ __device__
				Vector3ui operator()(MapVoxelID voxel_id)
			{
				return voxelmap::mapToVoxels(base_ptr, dim, base_ptr + voxel_id);
			}

			const Voxel* base_ptr;
			Vector3ui dim;
		};

		template<typename Voxel>
		struct voxelid_to_voxel
		{
			voxelid_to_voxel(Voxel* base_ptr)
				: base_ptr(base_ptr)
			{
			}

			__host__ __device__
				Voxel& operator()(MapVoxelID voxel_id)
			{
				return *(base_ptr + voxel_id);
			}

			Voxel* base_ptr;
		};

		template<typename Voxel>
		struct voxel_to_voxelid : public thrust::unary_function<const Voxel&, MapVoxelID>
		{
			voxel_to_voxelid(const Voxel* base_ptr)
				: base_ptr(base_ptr)
			{
			}

			__host__ __device__
				MapVoxelID operator()(const Voxel& voxel) const
			{
				return &voxel - base_ptr;
			}

			const Voxel* base_ptr;
		};

		template<>
		bool TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning)
		{
			std::scoped_lock lock(this->m_mutex, other->m_mutex);

			switch (other->getMapType())
			{
			case MT_BITVECTOR_VOXELLIST:
			{
				auto* m = other->as<BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID> >();

				const uint32_t num_new_voxels = m->getDimensions().x();
				const uint32_t offset_new_entries = m_dev_list.size();
				// resize capacity
				this->resize(offset_new_entries + num_new_voxels);

				// We append the given list to our own list of points.
				thrust::copy(
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_list.begin(), m->m_dev_coord_list.begin(), m->m_dev_id_list.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_list.end(), m->m_dev_coord_list.end(), m->m_dev_id_list.end())),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_list.begin() + offset_new_entries, m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				// if a new meaning was given, iterate over the voxellist and overwrite the meaning
				if (new_meaning)
				{
					BitVectorVoxel fillVoxel;
					fillVoxel.bitVector().setBit(*new_meaning);
					thrust::fill(m_dev_list.begin() + offset_new_entries, m_dev_list.end(), fillVoxel);
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				make_unique();

				return true;
			}
			case MT_COUNTING_VOXELLIST:
			{
				auto* m = other->as<CountingVoxelList>();

				const uint32_t num_new_voxels = m->getDimensions().x();
				const uint32_t offset_new_entries = m_dev_list.size();
				// resize capacity
				this->resize(offset_new_entries + num_new_voxels);

				// We append the given list to our own list of points.
				thrust::copy(
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_coord_list.begin(), m->m_dev_id_list.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_coord_list.end(), m->m_dev_id_list.end())),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				BitVectorVoxel fillVoxel;
				if (new_meaning)
				{
					fillVoxel.bitVector().setBit(*new_meaning);
				}
				// iterate over the voxellist and overwrite the meaning
				thrust::fill(m_dev_list.begin() + offset_new_entries, m_dev_list.end(), fillVoxel);
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				make_unique();

				return true;
			}
			case MT_BITVECTOR_VOXELMAP:
			{
				auto m = std::static_pointer_cast<voxelmap::BitVectorVoxelMap>(other);
				
				std::scoped_lock lock(this->m_mutex, m->m_mutex);

				const size_t offset_new_entries = m_dev_list.size();

				// Resize list to add space for new voxels
				const size_t num_new_entries = thrust::count_if(
					thrust::device_ptr<BitVectorVoxel>(m->getDeviceDataPtr()),
					thrust::device_ptr<BitVectorVoxel>(m->getDeviceDataPtr() + m->getVoxelMapSize()),
					is_occupied<BitVectorVoxel>(0.0f)); // Threshold doesn't matter for BitVectors
				this->resize(offset_new_entries + num_new_entries);

				// Fill MapVoxelIDs of occupied voxels into end of m_dev_id_list
				thrust::copy_if(
					thrust::counting_iterator<MapVoxelID>(0),                     // src.begin
					thrust::counting_iterator<MapVoxelID>(m->getVoxelMapSize()),  // src.end
					thrust::device_ptr<BitVectorVoxel>(m->getDeviceDataPtr()),    // stencil.begin (predicate is used here)
					m_dev_id_list.begin() + offset_new_entries,                   // dest.begin
					is_occupied<BitVectorVoxel>(0.0f)); // Threshold doesn't matter for BitVectors
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// Fill m_dev_coord_list and m_dev_list by transforming added values in m_dev_id_list
				thrust::transform(
					m_dev_id_list.begin() + offset_new_entries,
					m_dev_id_list.end(),
					m_dev_coord_list.begin() + offset_new_entries,
					voxelid_to_voxelcoord<BitVectorVoxel>(m->getDeviceDataPtr(), m->getDimensions()));
				thrust::transform(
					m_dev_id_list.begin() + offset_new_entries,
					m_dev_id_list.end(),
					m_dev_list.begin() + offset_new_entries,
					voxelid_to_voxel<BitVectorVoxel>(m->getDeviceDataPtr()));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				// if a new meaning was given, iterate over the voxellist and overwrite the meaning
				if (new_meaning)
				{
					BitVectorVoxel fillVoxel;
					fillVoxel.bitVector().setBit(*new_meaning);
					thrust::fill(m_dev_list.begin() + offset_new_entries, m_dev_list.end(), fillVoxel);
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				make_unique();
				return true;
			}
			default:
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
				return false;
			}
			}
		}

		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::merge(const GpuVoxelsMapSharedPtr other, const Vector3f& metric_offset, const BitVoxelMeaning* new_meaning)
		{
			const Vector3i voxel_offset = voxelmap::mapToVoxelsSigned(m_voxel_side_length, metric_offset);
			return merge(other, voxel_offset, new_meaning); // does the locking
		}

		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::subtract(const TemplateVoxelList<Voxel, VoxelIDType>* other, const Vector3i& voxel_offset)
		{
			std::scoped_lock lock(this->m_mutex, other->m_mutex);

			// find the overlapping voxels:
			thrust::device_vector<bool> overlap_stencil(m_dev_id_list.size()); // A stencil of the voxels in collision
			collideVoxellists(other, voxel_offset, overlap_stencil);

			keyCoordVoxelZipIterator new_end;


			// remove the overlapping voxels:
			new_end = thrust::remove_if(this->getBeginTripleZipIterator(),
				this->getEndTripleZipIterator(),
				overlap_stencil.begin(),
				thrust::identity<bool>());
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			const size_t new_length = thrust::distance(m_dev_id_list.begin(), thrust::get<0>(new_end.get_iterator_tuple()));
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			this->resize(new_length);

			return true;
		}

		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::subtract(const TemplateVoxelList<Voxel, VoxelIDType>* other, const Vector3f& metric_offset)
		{
			const Vector3i voxel_offset = voxelmap::mapToVoxelsSigned(m_voxel_side_length, metric_offset);
			return subtract(other, voxel_offset); // does the locking
		}


		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::subtractFromCountingVoxelList(const TemplateVoxelList<BitVectorVoxel, VoxelIDType>* other, const Vector3i& voxel_offset)
		{
			std::scoped_lock lock(this->m_mutex, other->m_mutex);

			// find the overlapping voxels:
			thrust::device_vector<bool> overlap_stencil(m_dev_id_list.size()); // A stencil of the voxels in collision
			collideVoxellists(other, voxel_offset, overlap_stencil);

			keyCoordVoxelZipIterator new_end;


			// remove the overlapping voxels:
			new_end = thrust::remove_if(this->getBeginTripleZipIterator(),
				this->getEndTripleZipIterator(),
				overlap_stencil.begin(),
				thrust::identity<bool>());
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			const size_t new_length = thrust::distance(m_dev_id_list.begin(), thrust::get<0>(new_end.get_iterator_tuple()));
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			this->resize(new_length);

			return true;
		}

		template<class Voxel, class VoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::subtractFromCountingVoxelList(const TemplateVoxelList<BitVectorVoxel, VoxelIDType>* other, const Vector3f& metric_offset)
		{
			const Vector3i voxel_offset = voxelmap::mapToVoxelsSigned(m_voxel_side_length, metric_offset);
			return subtractFromCountingVoxelList(other, voxel_offset); // does the locking
		}


		template<class Voxel, class VoxelIDType>
		Vector3ui TemplateVoxelList<Voxel, VoxelIDType>::getDimensions() const
		{
			//LOGGING_DEBUG_C(VoxellistLog, TemplateVoxelList, "This returns the number of voxels in the voxellist, not the xyz limits of the reference GpuVoxelsMap! The x value contains the number of voxels in the list." << endl);
			return Vector3ui(m_dev_list.size(), 1, 1);
		}

		template<class Voxel, class VoxelIDType>
		Vector3f TemplateVoxelList<Voxel, VoxelIDType>::getMetricDimensions() const
		{
			return Vector3f(m_ref_map_dim.x(), m_ref_map_dim.y(), m_ref_map_dim.z()) * getVoxelSideLength();
		}

		template<class Voxel, class VoxelIDType>
		TemplateVoxelList<Voxel, VoxelIDType>::keyCoordVoxelZipIterator TemplateVoxelList<Voxel, VoxelIDType>::getBeginTripleZipIterator()
		{
			return thrust::make_zip_iterator(thrust::make_tuple(m_dev_id_list.begin(), m_dev_coord_list.begin(), m_dev_list.begin()));
		}

		template<class Voxel, class VoxelIDType>
		TemplateVoxelList<Voxel, VoxelIDType>::keyCoordVoxelZipIterator TemplateVoxelList<Voxel, VoxelIDType>::getEndTripleZipIterator()
		{
			return thrust::make_zip_iterator(thrust::make_tuple(m_dev_id_list.end(), m_dev_coord_list.end(), m_dev_list.end()));
		}


		template <class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::extractCubes(thrust::device_vector<Cube>** output_vector) const
		{
			std::lock_guard guard(this->m_mutex);

			try
			{
				if (*output_vector == nullptr)
				{
					*output_vector = new thrust::device_vector<Cube>(m_dev_list.size());
				}
				else
				{
					(*output_vector)->resize(m_dev_list.size());
				}
				// Transform Iterator that takes coordinates and bitvector and writes cubes to output_vector
				thrust::transform(m_dev_coord_list.begin(), m_dev_coord_list.end(), m_dev_list.begin(), (*output_vector)->begin(),
					VoxelToCube());
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			}
			catch (thrust::system_error& e)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception " << e.what() << endl);
				exit(-1);
			}
		}


		template <class Voxel, class VoxelIDType>
		template<class OtherVoxel, class Collider>
		size_t TemplateVoxelList<Voxel, VoxelIDType>::collisionCheckWithCollider(const voxelmap::TemplateVoxelMap<OtherVoxel>* other,
			Collider collider, const Vector3i& offset)
		{
			// Map Dims have to be equal to be able to compare pointer adresses!
			if (other->getDimensions() != m_ref_map_dim)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList,
					"The dimensions of the Voxellist reference map do not match the colliding voxel map dimensions. Not checking collisions!" << endl);
				return (std::numeric_limits<size_t>::max)();
			}


			std::scoped_lock lock(this->m_mutex, other->m_mutex);

			// get raw pointers to the thrust vectors data:
			Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
			VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

			uint32_t num_blocks, threads_per_block;
			computeLinearLoad(getDimensions().x(), num_blocks, threads_per_block);
			size_t dynamic_shared_mem_size = sizeof(BitVectorVoxel) * cMAX_THREADS_PER_BLOCK;

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			kernelCollideWithVoxelMap<<<num_blocks, threads_per_block, dynamic_shared_mem_size>>>(dev_id_list_ptr, dev_voxel_list_ptr, getDimensions().x(),
				other->getConstDeviceDataPtr(), m_ref_map_dim, collider, offset,
				m_dev_collision_check_results_counter.data().get());
			CHECK_CUDA_ERROR();
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			m_collision_check_results_counter = m_dev_collision_check_results_counter;
			
			uint32_t number_of_collisions = 0;
			for (uint32_t i = 0; i < num_blocks; i++)
				number_of_collisions += m_collision_check_results_counter[i];

			return number_of_collisions;
		}

		template<class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::screendump(bool with_voxel_content) const
		{

			LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "Dumping VoxelList with " << getDimensions().x() << " entries to screen: " << endl);

			std::lock_guard guard(this->m_mutex);

			thrust::host_vector<VoxelIDType> host_id_list = m_dev_id_list;
			thrust::host_vector<Vector3ui> host_coord_list = m_dev_coord_list;
			thrust::host_vector<Voxel> host_list = m_dev_list;

			if (with_voxel_content)
			{
				for (uint32_t i = 0; i < host_list.size(); i++)
				{
					std::cout << "[" << i << "] ID = " << host_id_list[i] << " Coords: " << host_coord_list[i] << "Voxel: " << host_list[i] << std::endl;
				}
			}
			else {
				for (uint32_t i = 0; i < host_list.size(); i++)
				{
					std::cout << "[" << i << "] ID = " << host_id_list[i] << " Coords: " << host_coord_list[i] << std::endl;
				}
			}

			LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "Dumped " << getDimensions().x() << " Voxels." << endl);
		}


		template <class Voxel, class VoxelIDType>
		template<class OtherVoxel, class OtherVoxelIDType>
		bool TemplateVoxelList<Voxel, VoxelIDType>::equals(const TemplateVoxelList<OtherVoxel, OtherVoxelIDType>& other) const
		{
			if ((m_ref_map_dim != other.m_ref_map_dim) ||
				(m_voxel_side_length != other.m_voxel_side_length) ||
				(getDimensions()) != other.getDimensions())
			{
				return false;
			}
			std::scoped_lock lock(this->m_mutex, other.m_mutex);

			bool equal = true;
			equal &= thrust::equal(m_dev_list.begin(), m_dev_list.end(), other.m_dev_list.begin());
			equal &= thrust::equal(m_dev_id_list.begin(), m_dev_id_list.end(), other.m_dev_id_list.begin());
			equal &= thrust::equal(m_dev_coord_list.begin(), m_dev_coord_list.end(), other.m_dev_coord_list.begin());

			return equal;
		}

		template <class Voxel, class VoxelIDType>
		void TemplateVoxelList<Voxel, VoxelIDType>::clone(const TemplateVoxelList<Voxel, VoxelIDType>& other)
		{
			if (this->m_voxel_side_length != other.m_voxel_side_length || this->m_ref_map_dim != other.m_ref_map_dim)
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Voxellist cannot be cloned since map reference dimensions or voxel side length are not equal" << endl);
				return;
			}

			std::scoped_lock lock(this->m_mutex, other.m_mutex);

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			this->m_dev_id_list.resize(other.m_dev_id_list.size());
			this->m_dev_coord_list.resize(other.m_dev_coord_list.size());
			this->m_dev_list.resize(other.m_dev_list.size());

			thrust::copy(other.m_dev_id_list.begin(), other.m_dev_id_list.end(), this->m_dev_id_list.begin());
			thrust::copy(other.m_dev_coord_list.begin(), other.m_dev_coord_list.end(), this->m_dev_coord_list.begin());
			thrust::copy(other.m_dev_list.begin(), other.m_dev_list.end(), this->m_dev_list.begin());
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
		}

	} // end of namespace voxellist
} // end of namespace gpu_voxels
#endif