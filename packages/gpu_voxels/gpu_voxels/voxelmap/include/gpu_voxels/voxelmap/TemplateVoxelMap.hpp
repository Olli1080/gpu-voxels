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
#ifndef GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_HPP_INCLUDED

#include "TemplateVoxelMap.h"

#include <iostream>
#include <fstream>
#include <cfloat>

#include <gpu_voxels/logging/logging_voxelmap.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.hpp>
#include <gpu_voxels/voxel/DefaultCollider.hpp>
#include <gpu_voxels/voxel/SVCollider.hpp>
#include <gpu_voxels/voxellist/TemplateVoxelList.h>

#include <gpu_voxels/helpers/PointCloud.h>
#include <gpu_voxels/helpers/MathHelpers.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace gpu_voxels
{
	namespace voxelmap
	{

		//#define ALTERNATIVE_CHECK
#undef  ALTERNATIVE_CHECK

#ifdef  ALTERNATIVE_CHECK
#define LOOP_SIZE       4
#endif



//const uint32_t cMAX_POINTS_PER_ROBOT_SEGMENT = 118000;

		template<class Voxel>
		TemplateVoxelMap<Voxel>::TemplateVoxelMap(const Vector3ui& dim,
			const float voxel_side_length, const MapType map_type) :
			m_dim(dim),
			m_limits(dim.x() * voxel_side_length, dim.y() * voxel_side_length, dim.z() * voxel_side_length),
			m_voxel_side_length(voxel_side_length),
			m_collision_check_results(cMAX_NR_OF_BLOCKS, false),
			m_collision_check_results_counter(cMAX_NR_OF_BLOCKS, 0),
			m_dev_data(getVoxelMapSize()),
			m_dev_points_outside_map(nullptr)
		{
			this->m_map_type = map_type;
			if (dim.x() * dim.y() * dim.z() * sizeof(Voxel) > (pow(2, 32) - 1))
			{
				LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Map size limited to 32 bit addressing!" << endl);
				exit(-1);
			}

			if (getVoxelMapSize() * sizeof(Voxel) > (pow(2, 32) - 1))
			{
				LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Memory size is limited to 32 bit!" << endl);
				exit(-1);
			}
			HANDLE_CUDA_ERROR(cudaEventCreate(&m_start));
			HANDLE_CUDA_ERROR(cudaEventCreate(&m_stop));

			// the voxelmap

			//HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_data, getMemoryUsage()));
			LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "Voxelmap base address is " << (void*)m_dev_data.data().get() << endl);

			HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_points_outside_map, sizeof(bool)));


			computeLinearLoad(m_dev_data.size(), m_blocks, m_threads);
#ifdef ALTERNATIVE_CHECK
			computeLinearLoad((uint32_t)ceil((float)m_dev_data.size() / (float)LOOP_SIZE), &m_alternative_blocks, &m_alternative_threads);
#endif

			m_dev_collision_check_results = m_collision_check_results;
			m_dev_collision_check_results_counter = m_collision_check_results_counter;
			
			clearMap();

#ifndef ALTERNATIVE_CHECK
			// determine size of array for results of collision check
			m_result_array_size = std::min(static_cast<uint32_t>(m_dev_data.size()), cMAX_NR_OF_BLOCKS);
			
#else

			// determine size of array for results of collision check
			if (ceil((float)m_dev_data.size() / (float)LOOP_SIZE) >= cMAX_NR_OF_BLOCKS)
			{
				m_result_array_size = cMAX_NR_OF_BLOCKS;
			}
			else
			{
				m_result_array_size = ceil((float)m_dev_data.size() / ((float)cMAX_THREADS_PER_BLOCK * (float)LOOP_SIZE));
			}

#endif

		}
		template<class Voxel>
		TemplateVoxelMap<Voxel>::TemplateVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
			m_dim(dim), m_limits(dim.x() * voxel_side_length, dim.y() * voxel_side_length,dim.z() * voxel_side_length),
			m_voxel_side_length(voxel_side_length), m_dev_data(getVoxelMapSize()) //TODO:: here is a bug!!
		{
			this->m_map_type = map_type;

			computeLinearLoad(m_dev_data.size(), m_blocks, m_threads);
		}

		template<class Voxel>
		TemplateVoxelMap<Voxel>::~TemplateVoxelMap()
		{
			if (m_dev_points_outside_map)
				HANDLE_CUDA_ERROR(cudaFree(m_dev_points_outside_map));

			HANDLE_CUDA_ERROR(cudaEventDestroy(m_start));
			HANDLE_CUDA_ERROR(cudaEventDestroy(m_stop));
		}

		/* ======== VoxelMap operations  ======== */

		/*!
		 * Specialized clearing function for Bitmap Voxelmaps.
		 * As the bitmap for every voxel is empty,
		 * it is sufficient to set the whole map to zeroes.
		 * WATCH OUT: This sets the map to eBVM_FREE and not to eBVM_UNKNOWN!
		 */
		template<>
		inline void TemplateVoxelMap<BitVectorVoxel>::clearMap()
		{
			std::lock_guard guard(this->m_mutex);
			// Clear occupancies
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			thrust::fill(thrust::device, m_dev_data.begin(), m_dev_data.end(), gpu_voxels::BitVectorVoxel());

			// Clear result array
			for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
				m_collision_check_results[i] = false;

			m_dev_collision_check_results = m_collision_check_results;
		}

		/*!
		 * Specialized clearing function for probabilistic Voxelmaps.
		 * As a ProbabilisticVoxel consists of only one byte, we can
		 * memset the whole map to UNKNOWN_PROBABILITY
		 */
		template<>
		inline void TemplateVoxelMap<ProbabilisticVoxel>::clearMap()
		{
			std::lock_guard guard(this->m_mutex);
			// Clear occupancies
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			thrust::fill(m_dev_data.begin(), m_dev_data.end(), UNKNOWN_PROBABILITY);

			// Clear result array
			for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
				m_collision_check_results[i] = false;

			m_dev_collision_check_results = m_collision_check_results;
		}

		/*!
		 * Specialized clearing function for DistanceVoxelmaps.
		 * it is sufficient to set the whole map to PBA_UNINITIALISED_COORD.
		 */
		template<>
		inline void TemplateVoxelMap<DistanceVoxel>::clearMap()
		{
			std::lock_guard guard(this->m_mutex);
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			//  //deprecated: initialising voxels to all zero
			//  HANDLE_CUDA_ERROR(cudaMemset(m_dev_data, 0, m_dev_data.size()*sizeof(DistanceVoxel)));

			// Clear contents: distance of PBA_UNINITIALISED indicates uninitialized voxel
			DistanceVoxel pba_uninitialised_voxel;
			pba_uninitialised_voxel.setPBAUninitialised();

			thrust::fill(m_dev_data.begin(), m_dev_data.end(), pba_uninitialised_voxel);

			//  //TODO: adapt for distanceVoxel? eliminate?
			//  // Clear result array
			//  for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
			//  {
			//    m_collision_check_results[i] = false;
			//  }
			//  HANDLE_CUDA_ERROR(
			//      cudaMemcpy(m_dev_collision_check_results, m_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
			//                 cudaMemcpyHostToDevice));

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
		}

		template<>
		inline void TemplateVoxelMap<CountingVoxel>::clearMap()
		{
			std::lock_guard guard(this->m_mutex);
			// Clear occupancies
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			thrust::fill(m_dev_data.begin(), m_dev_data.end(), gpu_voxels::CountingVoxel());

			// Clear result array
			for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
				m_collision_check_results[i] = false;

			m_dev_collision_check_results = m_collision_check_results;
		}

		struct sumVector3ui
		{
			__host__ __device__
			sumVector3ui() : sum(), count()
			{
			}

			__host__ __device__
			sumVector3ui(Vector3ui sum, uint32_t count) : sum(sum), count(count)
			{
			}

			__host__ __device__
			sumVector3ui operator()(sumVector3ui a, sumVector3ui b)
			{
				return { a.sum + b.sum, a.count + b.count };
			}

			Vector3ui sum = Vector3ui::Zero();
			uint32_t count;
		};

		template<class Voxel>
		struct toCoordsIfInBoundsAndOccupied
		{
			Vector3ui upper_bound;
			Vector3ui lower_bound;
			Vector3ui dim;

			toCoordsIfInBoundsAndOccupied(Vector3ui upper, Vector3ui lower, Vector3ui dimensions) :
				upper_bound(upper), lower_bound(lower), dim(dimensions)
			{
			}

			__host__ __device__
				sumVector3ui operator()(thrust::tuple<Voxel, int> t)
			{
				Voxel v = thrust::get<0>(t);
				Vector3ui a = mapToVoxels(thrust::get<1>(t), dim);

				bool inBound = ((a.x() >= lower_bound.x()) && (a.y() >= lower_bound.y()) && (a.z() >= lower_bound.z())
					&& (a.x() < upper_bound.x()) && (a.y() < upper_bound.y()) && (a.z() < upper_bound.z()));

				return (inBound && v.isOccupied(0.0f)) ? sumVector3ui(a, 1) : sumVector3ui(); //return 0/0/0 if not occupied and in bounds
			}
		};

		template<>
		inline Vector3f TemplateVoxelMap<DistanceVoxel>::getCenterOfMass() const
		{
			LOGGING_INFO_C(VoxelmapLog, TemplateVoxelMap, "Center of Mass is not implemented for DistanceVoxel yet, because the it doesnt has IsOccupied" << endl);
			return {};
		}

		template<class Voxel>
		Vector3f TemplateVoxelMap<Voxel>::getCenterOfMass() const
		{
			return getCenterOfMass(Vector3ui::Zero(), m_dim);
		}

		template<>
		inline Vector3f TemplateVoxelMap<DistanceVoxel>::getCenterOfMass(Vector3ui lower_bound, Vector3ui upper_bound) const
		{
			LOGGING_INFO_C(VoxelmapLog, TemplateVoxelMap, "Center of Mass is not implemented for DistanceVoxel yet, because the it doesnt has IsOccupied" << endl);
			return {};
		}

		template<class Voxel>
		Vector3f TemplateVoxelMap<Voxel>::getCenterOfMass(Vector3ui lower_bound, Vector3ui upper_bound) const
		{
			//filter by axis aligned bounding box
			uint32_t boundary_voxel_count = (upper_bound.x() - lower_bound.x())
				* (upper_bound.y() - lower_bound.y()) * (upper_bound.z() - lower_bound.z());
			if (boundary_voxel_count <= 0)
			{
				LOGGING_INFO_C(VoxelmapLog, TemplateVoxelMap, "No Voxels Found in given Bounding Box: Lower Bound ("
					<< lower_bound.x() << "|" << lower_bound.y() << "|" << lower_bound.z() << ")  "
					<< "Upper Bound (" << upper_bound.x() << "|" << upper_bound.y() << "|" << upper_bound.z() << ")" << endl);

				return {};
			}

			toCoordsIfInBoundsAndOccupied<Voxel> filter(upper_bound, lower_bound, m_dim);
			typedef thrust::counting_iterator<int> count_it;

			const sumVector3ui sumVector = thrust::transform_reduce(
				thrust::make_zip_iterator(thrust::make_tuple(m_dev_data.begin(), count_it(0))),
				thrust::make_zip_iterator(thrust::make_tuple(m_dev_data.end(), count_it(m_dev_data.size()))),
				filter, sumVector3ui(Vector3ui::Zero(), 0), sumVector3ui());
			
			//divide by voxel count
			const Vector3f metricSum = sumVector.sum.template cast<float>() * m_voxel_side_length;
			const uint32_t voxelCount = sumVector.count;
			const Vector3f coM = metricSum / voxelCount;;
			return coM;
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::printVoxelMapData()
		{
			std::lock_guard guard(this->m_mutex);
			HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_data.data().get(), m_dev_data.size(), "VoxelMap dump: "));
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::gatherVoxelsByIndex(thrust::device_ptr<unsigned int> dev_indices_begin, thrust::device_ptr<unsigned int> dev_indices_end, thrust::device_ptr<Voxel> dev_output_begin)
		{
			// writes the Voxels at indicated indices to dev_output_begin
			thrust::gather(dev_indices_begin, dev_indices_end,
				thrust::device_pointer_cast(m_dev_data.data()),
				dev_output_begin);
		}

		//template<class Voxel>
		//bool TemplateVoxelMap<Voxel>::collisionCheckAlternative(const uint8_t threshold, VoxelMap* other,
		//                                         const uint8_t other_threshold, uint32_t loop_size)
		//{
		// Todo: DO LOCKING HERE!!
		//  computeLinearLoad((uint32_t) ceil((float) m_dev_data.size() / (float) loop_size),
		//                           &m_alternative_blocks, &m_alternative_threads);
		////  printf("number of blocks: %i , number of threads: %i", m_alternative_blocks, m_alternative_threads);
		//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
		//  m_elapsed_time = 0;
		////  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
		//  kernelCollideVoxelMapsAlternative<<< m_alternative_blocks, m_alternative_threads >>>
		//  (m_dev_data, m_dev_data.size(), threshold, other->getDeviceDataPtr(), other_threshold, loop_size, m_dev_collision_check_results);
		//  CHECK_CUDA_ERROR();
		////  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
		//  HANDLE_CUDA_ERROR(
		//      cudaMemcpy(m_collision_check_results, m_dev_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
		//                 cudaMemcpyDeviceToHost));
		//
		////  bool test;
		//  for (uint32_t i = 0; i < m_result_array_size; i++)
		//  {
		//    // collision as soon as first result is true
		//    if (m_collision_check_results[i])
		//    {
		////      HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
		////      HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
		////      HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
		////      printf(" ...done in %f ms!\n", m_elapsed_time);
		////      m_measured_data.push_back(m_elapsed_time);
		//      return true;
		//    }
		//  }
		////  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
		////  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
		////  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
		////  printf(" ...done in %f ms!\n", m_elapsed_time);
		////  m_measured_data.push_back(m_elapsed_time);
		//  //HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_collision_check_results, cMAX_NR_OF_BLOCKS, " collision array on device "));
		//  return false;
		//}
		template<class Voxel>
		template<class OtherVoxel, class Collider>
		bool TemplateVoxelMap<Voxel>::collisionCheck(TemplateVoxelMap<OtherVoxel>* other, Collider collider)
		{
			std::scoped_lock lock(this->m_mutex, other->m_mutex);
			//printf("collision check... ");

#ifndef ALTERNATIVE_CHECK
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			//  m_elapsed_time = 0;
			//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
			//  printf("TemplateVoxelMap<Voxel>::collisionCheck\n");

			kernelCollideVoxelMaps<<<m_blocks, m_threads>>>(m_dev_data.data().get(), m_dev_data.size(), other->getDeviceDataPtr(),
				collider, m_dev_collision_check_results);
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			m_collision_check_results = m_dev_collision_check_results;

			for (uint32_t i = 0; i < m_blocks; i++)
			{
				//printf(" results[%d] = %s\n", i, m_collision_check_results[i]? "collision" : "no collision");
				// collision as soon as first result is true
				if (m_collision_check_results[i])
				{
					//      HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
					//      HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
					//      HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
					//      printf(" ...done in %f ms!\n", m_elapsed_time);
					//      m_measured_data.push_back(m_elapsed_time);
					//      printf("TemplateVoxelMap<Voxel>::collisionCheck finished\n");
					return true;
				}
			}
			//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
			//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
			//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
			  //printf(" ...done in %f ms!\n", m_elapsed_time);
			  //HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_collision_check_results, cMAX_NR_OF_BLOCKS, " collision array on device "));
			//  m_measured_data.push_back(m_elapsed_time);
			//  printf("TemplateVoxelMap<Voxel>::collisionCheck finished\n");

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			return false;

#else

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			m_elapsed_time = 0;
			//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
			kernelCollideVoxelMapsAlternative << < m_alternative_blocks, m_alternative_threads >> >
				(m_dev_data.data().get(), m_dev_data.size(), threshold, other->getDeviceDataPtr(), other_threshold, LOOP_SIZE, m_dev_collision_check_results);
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			HANDLE_CUDA_ERROR(cudaMemcpy(m_collision_check_results, m_dev_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool), cudaMemcpyDeviceToHost));

			for (uint32_t i = 0; i < m_result_array_size; i++)
			{
				// collision as soon as first result is true
				if (m_collision_check_results[i])
				{
					//      HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
					//      HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
					//      HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
					//      printf(" ...done in %f ms!\n", m_elapsed_time);
					//      m_measured_data.push_back(m_elapsed_time);
					return true;
				}
			}
			//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
			//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
			//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
			//  printf(" ...done in %f ms!\n", m_elapsed_time);
			//  m_measured_data.push_back(m_elapsed_time);
			  //HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_collision_check_results, cMAX_NR_OF_BLOCKS, " collision array on device "));
			return false;

#endif
		}

		//template<class Voxel>
		//bool TemplateVoxelMap<Voxel>::collisionCheckBoundingBox(uint8_t threshold, VoxelMap* other, uint8_t other_threshold,
		//                                         Vector3ui bounding_box_start, Vector3ui bounding_box_end)
		//{
		//  int number_of_blocks = bounding_box_end.y - bounding_box_start.y;
		////	number_of_blocks = number_of_blocks * (bounding_box_end.x - bounding_box_start.x);
		//  int number_of_threads = bounding_box_end.z - bounding_box_start.z;
		//  int number_of_thread_runs = bounding_box_end.x - bounding_box_start.x;
		//
		//  bool* dev_result;
		//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
		//  //aquiring memory for results:
		//  HANDLE_CUDA_ERROR(cudaMalloc((void** )&dev_result, sizeof(bool) * number_of_blocks * number_of_threads));
		//
		//  kernelCollideVoxelMapsBoundingBox<<<number_of_blocks, number_of_threads, number_of_threads>>>
		//  (m_dev_data, m_dev_data.size(), threshold, other->m_dev_data, other_threshold,
		//      dev_result, bounding_box_start.x, bounding_box_start.y, bounding_box_start.z,
		//      number_of_thread_runs, m_dim);
		//  CHECK_CUDA_ERROR();
		//
		//  bool result_array[number_of_blocks * number_of_threads];
		//  //Copying results back
		//  HANDLE_CUDA_ERROR(
		//      cudaMemcpy(&result_array[0], dev_result, sizeof(bool) * number_of_blocks * number_of_threads,
		//                 cudaMemcpyDeviceToHost));
		//
		//  bool result = false;
		//  //reducing result
		//
		//  for (int u = 0; u < number_of_blocks * number_of_threads; ++u)
		//  {
		////		printf("coll-check-bounding: block nr %i , result %i \n", u, result_array[u]);
		//    if (result_array[u])
		//    {
		//      LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "Collision occurred!" << endl);
		//      result = true;
		//      break;
		//    }
		//  }
		//  LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "No Collision occurred!" << endl);
		//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
		//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
		//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
		////	      printf(" ...done in %f ms!\n", m_elapsed_time);
		//
		//  //releasing memory
		//
		//  HANDLE_CUDA_ERROR(cudaFree(dev_result));
		//
		//  return result;
		//  //
		//  //
		//  //
		//  //	void kernelCollideVoxelMapsBoundingBox(Voxel* voxelmap, const uint32_t voxelmap_size,
		//  //	                            const uint8_t threshold, Voxel* other_map,
		//  //	                            const uint8_t other_threshold, bool* results,
		//  //	                            uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
		//  //	                            uint32_t size_x, Vector3ui* dimensions)
		//
		//}

		template<class Voxel>
		template<class OtherVoxel, class Collider>
		uint32_t TemplateVoxelMap<Voxel>::collisionCheckWithCounter(TemplateVoxelMap<OtherVoxel>* other,
			Collider collider)
		{
			return collisionCheckWithCounterRelativeTransform(other, collider); //does the locking
		}


		template<class Voxel>
		template<class OtherVoxel, class Collider>
		uint32_t TemplateVoxelMap<Voxel>::collisionCheckWithCounterRelativeTransform(const TemplateVoxelMap<OtherVoxel>* other,
			Collider collider, const Vector3i& offset)
		{
			std::scoped_lock lock(this->m_mutex, other->m_mutex);

			Voxel* dev_data_with_offset = nullptr;
			if (offset != Vector3i::Zero())
			{
				// We take the base adress of this voxelmap and add the offset that we want to shift the other map.
				dev_data_with_offset = getVoxelPtrSignedOffset(m_dev_data.data().get(), m_dim, offset);
			}
			else 
			{
				dev_data_with_offset = m_dev_data.data().get();
			}
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			kernelCollideVoxelMapsDebug<<<m_blocks, m_threads>>>(dev_data_with_offset, m_dev_data.size(), other->getConstDeviceDataPtr(),
				collider, m_dev_collision_check_results_counter.data().get());
			CHECK_CUDA_ERROR();
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			m_collision_check_results_counter = m_dev_collision_check_results_counter;

			uint32_t number_of_collisions = 0;
			for (uint32_t i = 0; i < m_blocks; i++)
				number_of_collisions += m_collision_check_results_counter[i];

			return number_of_collisions;
		}


		//template<class Voxel>
		//void TemplateVoxelMap<Voxel>::copyVoxelMapDifferentSize(VoxelMap* destination, VoxelMap* source, bool with_bitvector)
		//{
		//  //factor between the voxel side length of the destination and the source map.
		//  //the Destination Map must alwas have a bigger or equal voxel side length as the
		//  //source map
		//  uint8_t factor = (uint8_t) (destination->getVoxelSideLength() / source->getVoxelSideLength());
		//
		//  uint32_t blocks = 0;
		//  uint32_t threads = 0;
		//
		//  destination->computeLinearLoad(destination->m_dev_data.size(), &blocks, &threads);
		//  if (with_bitvector)
		//  {
		//    kernelShrinkCopyVoxelMapBitvector<<<blocks, threads>>>(destination->m_dev_data,
		//                                                           destination->m_dev_data.size(),
		//                                                           destination->m_dim, source->m_dev_data,
		//                                                           source->m_dev_data.size(), source->m_dim,
		//                                                           factor);
		//    CHECK_CUDA_ERROR();
		//  }
		//  else
		//  {
		//    kernelShrinkCopyVoxelMap<<<blocks, threads>>>(destination->m_dev_data, destination->m_dev_data.size(),
		//                                                  destination->m_dim, source->m_dev_data,
		//                                                  source->m_dev_data.size(), source->m_dim, factor);
		//    CHECK_CUDA_ERROR();
		//
		//  }
		//
		//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
		////		void copyVoxelMapBitvector(Voxel* destination_map, const uint32_t destination_map_size, Vector3ui* dest_map_dim,
		////				Voxel* source_map, const uint32_t source_map_size, Vector3ui* source_map_dim, uint8_t factor)
		//}

		// ------ BEGIN Global API functions ------

		/**
		 * author: Matthias Wagner
		 * Inserts a voxel at each point from the points list.
		 */
		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertPointCloud(const std::vector<Vector3f>& points, const BitVoxelMeaning voxel_meaning)
		{
			// copy points to the gpu
			std::lock_guard guard(this->m_mutex);

			const thrust::device_vector<Vector3f> d_points = { points.begin(), points.end() };

			insertPointCloud(d_points, voxel_meaning);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertPointCloud(const PointCloud& pointcloud, const BitVoxelMeaning voxel_meaning)
		{
			std::lock_guard guard(this->m_mutex);

			insertPointCloud(pointcloud.getPointsDevice(), voxel_meaning);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertPointCloud(const thrust::device_vector<Vector3f>& d_points, const BitVoxelMeaning voxel_meaning)
		{
			// reset warning indicator:
			HANDLE_CUDA_ERROR(cudaMemset((void*)m_dev_points_outside_map, 0, sizeof(bool)));
			bool points_outside_map;

			uint32_t num_blocks, threads_per_block;
			computeLinearLoad(d_points.size(), num_blocks, threads_per_block);
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			kernelInsertGlobalPointCloud<<<num_blocks, threads_per_block>>>(m_dev_data.data().get(), m_dim, m_voxel_side_length,
				d_points.data().get(), d_points.size(), voxel_meaning, m_dev_points_outside_map);
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			if (points_outside_map)
				LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertCoordinateList(const std::vector<Vector3ui>& coordinates, const BitVoxelMeaning voxel_meaning)
		{
			// copy points to the gpu
			std::lock_guard guard(this->m_mutex);
			const thrust::device_vector<Vector3ui> d_coordinates = { coordinates.begin(), coordinates.end() };
			insertCoordinateList(d_coordinates, voxel_meaning);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertCoordinateList(const thrust::device_vector<Vector3ui>& d_coordinates, const BitVoxelMeaning voxel_meaning)
		{
			// reset warning indicator:
			HANDLE_CUDA_ERROR(cudaMemset((void*)m_dev_points_outside_map, 0, sizeof(bool)));
			bool points_outside_map;

			uint32_t num_blocks, threads_per_block;
			computeLinearLoad(d_coordinates.size(), num_blocks, threads_per_block);
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			kernelInsertCoordinateTuples<<<num_blocks, threads_per_block>>>(m_dev_data.data().get(), m_dim, m_voxel_side_length,
				d_coordinates.data().get(), d_coordinates.size(), voxel_meaning, m_dev_points_outside_map);
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			if (points_outside_map)
				LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertDilatedCoordinateList(Voxel* d_dest_data, const thrust::device_vector<Vector3ui>& d_src_coordinates, const BitVoxelMeaning voxel_meaning)
		{
			// reset warning indicator:
			HANDLE_CUDA_ERROR(cudaMemset((void*)m_dev_points_outside_map, 0, sizeof(bool)));
			bool points_outside_map;

			uint32_t num_blocks, threads_per_block;
			computeLinearLoad(d_src_coordinates.size(), num_blocks, threads_per_block);
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			kernelInsertDilatedCoordinateTuples<<<num_blocks, threads_per_block>>>(d_dest_data, m_dim, d_src_coordinates.data().get(), d_src_coordinates.size(), voxel_meaning, m_dev_points_outside_map);
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			if (points_outside_map)
			{
				LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);
			}
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertDilatedCoordinateList(const std::vector<Vector3ui>& coordinates, const BitVoxelMeaning voxel_meaning)
		{
			const thrust::device_vector<Vector3ui> d_coordinates(coordinates);
			this->insertDilatedCoordinateList(d_coordinates, voxel_meaning);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertDilatedCoordinateList(const thrust::device_vector<Vector3ui> d_coordinates, const BitVoxelMeaning voxel_meaning)
		{
			insertDilatedCoordinateList(this->m_dev_data.data().get(), d_coordinates, voxel_meaning);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertClosedCoordinateList(const thrust::device_vector<Vector3ui>& d_coordinates, const BitVoxelMeaning voxel_meaning, float erode_threshold, float occupied_threshold, Voxel* d_buffer)
		{
			insertDilatedCoordinateList(d_buffer, d_coordinates, voxel_meaning);
			erode(this->m_dev_data.data().get(), d_buffer, erode_threshold, occupied_threshold);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertClosedCoordinateList(const std::vector<Vector3ui>& coordinates, const BitVoxelMeaning voxel_meaning, float erode_threshold, float occupied_threshold, TemplateVoxelMap<Voxel>& buffer)
		{
			const thrust::device_vector<Vector3ui> d_coordinates(coordinates);
			this->insertClosedCoordinateList(d_coordinates, voxel_meaning, erode_threshold, occupied_threshold, buffer);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertClosedCoordinateList(const thrust::device_vector<Vector3ui>& d_coordinates, const BitVoxelMeaning voxel_meaning, float erode_threshold, float occupied_threshold, TemplateVoxelMap<Voxel>& buffer)
		{
			this->insertClosedCoordinateList(d_coordinates, voxel_meaning, erode_threshold, occupied_threshold, buffer.m_dev_data.data().get());
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertClosedCoordinateList(const std::vector<Vector3ui>& coordinates, const BitVoxelMeaning voxel_meaning, float erode_threshold, float occupied_threshold)
		{
			const thrust::device_vector<Vector3ui> d_coordinates(coordinates);
			this->insertClosedCoordinateList(d_coordinates, voxel_meaning, erode_threshold, occupied_threshold);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertClosedCoordinateList(const thrust::device_vector<Vector3ui>& d_coordinates, const BitVoxelMeaning voxel_meaning, float erode_threshold, float occupied_threshold)
		{
			// Allocate temporary buffer
			thrust::device_vector<Voxel> d_buffer(this->getVoxelMapSize(), Voxel());
			this->insertClosedCoordinateList(d_coordinates, voxel_meaning, erode_threshold, occupied_threshold, d_buffer.data().get());
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::erode(Voxel* dest_data, const Voxel* src_data, float erode_threshold, float occupied_threshold) const
		{
			constexpr uint32_t BLOCK_SIDE_LENGTH = 8;
			dim3 num_blocks, threads_per_block;
			threads_per_block.x = BLOCK_SIDE_LENGTH;
			threads_per_block.y = BLOCK_SIDE_LENGTH;
			threads_per_block.z = BLOCK_SIDE_LENGTH;
			num_blocks.x = (m_dim.x() + threads_per_block.x - 1) / threads_per_block.x;
			num_blocks.y = (m_dim.y() + threads_per_block.y - 1) / threads_per_block.y;
			num_blocks.z = (m_dim.z() + threads_per_block.z - 1) / threads_per_block.z;

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			kernelErode<<<num_blocks, threads_per_block>>>(dest_data, src_data, this->m_dim, erode_threshold, occupied_threshold);
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			CHECK_CUDA_ERROR();
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::erodeInto(TemplateVoxelMap<Voxel>& dest, float erode_threshold, float occupied_threshold) const
		{
			assert(this->m_dim == dest.m_dim);
			erode(dest.m_dev_data.data().get(), this->m_dev_data.data().get(), erode_threshold, occupied_threshold);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::erodeLonelyInto(TemplateVoxelMap<Voxel>& dest, float occupied_threshold) const
		{
			this->erodeInto(dest, FLT_EPSILON, occupied_threshold);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertMetaPointCloud(const MetaPointCloud& meta_point_cloud,
			BitVoxelMeaning voxel_meaning)
		{
			std::lock_guard guard(this->m_mutex);

			// reset warning indicator:
			HANDLE_CUDA_ERROR(cudaMemset((void*)m_dev_points_outside_map, 0, sizeof(bool)));
			bool points_outside_map;

			computeLinearLoad(meta_point_cloud.getAccumulatedPointcloudSize(), m_blocks, m_threads);
			kernelInsertMetaPointCloud<<<m_blocks, m_threads>>>(
				m_dev_data.data().get(), meta_point_cloud.getDeviceConstPointer().get(), voxel_meaning, m_dim, m_voxel_side_length,
				m_dev_points_outside_map);
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
			if (points_outside_map)
				LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::insertMetaPointCloud(const MetaPointCloud& meta_point_cloud,
			const std::vector<BitVoxelMeaning>& voxel_meanings)
		{
			std::lock_guard guard(this->m_mutex);
			assert(meta_point_cloud.getNumberOfPointclouds() == voxel_meanings.size());

			// reset warning indicator:
			HANDLE_CUDA_ERROR(cudaMemset((void*)m_dev_points_outside_map, 0, sizeof(bool)));
			bool points_outside_map;

			computeLinearLoad(meta_point_cloud.getAccumulatedPointcloudSize(), m_blocks, m_threads);

			BitVoxelMeaning* voxel_meanings_d;
			const size_t size = voxel_meanings.size() * sizeof(BitVoxelMeaning);
			HANDLE_CUDA_ERROR(cudaMalloc(&voxel_meanings_d, size));
			HANDLE_CUDA_ERROR(cudaMemcpy(voxel_meanings_d, voxel_meanings.data(), size, cudaMemcpyHostToDevice));

			kernelInsertMetaPointCloud<<<m_blocks, m_threads>>>(
				m_dev_data.data().get(), meta_point_cloud.getDeviceConstPointer().get(), voxel_meanings_d, m_dim, m_voxel_side_length,
				m_dev_points_outside_map);
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
			if (points_outside_map)
				LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);

			HANDLE_CUDA_ERROR(cudaFree(voxel_meanings_d));
		}

		template<class Voxel>
		bool TemplateVoxelMap<Voxel>::writeToDisk(const std::string path)
		{
			std::lock_guard guard(this->m_mutex);
			std::ofstream out(path.c_str());
			if (!out.is_open())
			{
				LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Write to file " << path << " failed!" << endl);
				return false;
			}

			MapType map_type = this->getTemplateType();
			const auto buffer_size = static_cast<uint32_t>(this->getMemoryUsage());
			char* buffer = new char[buffer_size];

			LOGGING_INFO_C(VoxelmapLog, VoxelMap, "Dumping Voxelmap to disk: " <<
				getVoxelMapSize() << " Voxels ==> " << (buffer_size * cBYTE2MBYTE) << " MB. ..." << endl);

			HANDLE_CUDA_ERROR(cudaMemcpy(static_cast<void*>(buffer), this->getConstVoidDeviceDataPtr(), buffer_size, cudaMemcpyDeviceToHost));

			const bool bin_mode = true;
			// Write meta data and actual data
			if (bin_mode)
			{
				out.write((char*)&map_type, sizeof(MapType));
				out.write((char*)&m_voxel_side_length, sizeof(float));
				out.write((char*)&m_dim.x(), sizeof(uint32_t));
				out.write((char*)&m_dim.y(), sizeof(uint32_t));
				out.write((char*)&m_dim.z(), sizeof(uint32_t));
				out.write((char*)&buffer[0], buffer_size);
			}
			else
			{
				out << map_type << "\n";
				out << m_voxel_side_length << "\n";
				out << m_dim.x() << "\n";
				out << m_dim.y() << "\n";
				out << m_dim.z() << "\n";
				for (uint32_t i = 0; i < buffer_size; ++i)
					out << buffer[i] << "\n";
			}

			out.close();
			delete[] buffer;

			LOGGING_INFO_C(VoxelmapLog, VoxelMap, "... writing to disk is done." << endl);
			return true;
		}


		template<class Voxel>
		bool TemplateVoxelMap<Voxel>::readFromDisk(const std::string path)
		{
			std::lock_guard guard(this->m_mutex);
			MapType map_type;
			float voxel_side_length;
			uint32_t dim_x, dim_y, dim_z;


			std::ifstream in(path.c_str());
			if (!in.is_open())
			{
				LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Error in reading file " << path << endl);
				return false;
			}

			// Read meta data
			bool bin_mode = true;
			if (bin_mode)
			{
				in.read((char*)&map_type, sizeof(MapType));
				in.read((char*)&voxel_side_length, sizeof(float));
				in.read((char*)&dim_x, sizeof(uint32_t));
				in.read((char*)&dim_y, sizeof(uint32_t));
				in.read((char*)&dim_z, sizeof(uint32_t));
			}
			else
			{
				int tmp;
				in >> tmp;
				map_type = static_cast<MapType>(tmp);
				in >> voxel_side_length;
				in >> dim_x;
				in >> dim_y;
				in >> dim_z;
			}

			Vector3ui dim(dim_x, dim_y, dim_z);

			// Check meta data
			if (map_type != this->getTemplateType())
			{
				LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Voxelmap type does not match!" << endl);
				return false;
			}
			if (voxel_side_length != this->m_voxel_side_length)
			{
				LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "Read from file failed: Read Voxel side length (" << voxel_side_length <<
					") does not match current object (" << m_voxel_side_length << ")! Continuing though..." << endl);
			}
			else {
				if (map_type == MT_BITVECTOR_VOXELMAP)
				{
					//TODO: Check for size of bitvector, as that may change!
				}
			}
			if (dim != m_dim)
			{
				// after that check we may use the class size variables.
				LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Read from file failed: Read reference map dimension (" << dim << ") does not match current object (" << m_dim << ")!" << endl);

				return false;
			}

			// Read actual data
			LOGGING_INFO_C(VoxelmapLog, VoxelMap, "Reading Voxelmap from disk: " <<
				getVoxelMapSize() << " Voxels ==> " << (getMemoryUsage() * cBYTE2MBYTE) << " MB. ..." << endl);
			std::vector<char> buffer(getMemoryUsage());

			if (bin_mode)
			{
				in.read(buffer.data(), getMemoryUsage());
			}
			else {
				for (uint32_t i = 0; i < getMemoryUsage(); ++i)
					in >> buffer[i];
			}

			// Copy data to device
			HANDLE_CUDA_ERROR(cudaMemcpy(this->getVoidDeviceDataPtr(), buffer.data(), getMemoryUsage(), cudaMemcpyHostToDevice));

			in.close();
			LOGGING_INFO_C(VoxelmapLog, VoxelMap, "... reading from disk is done." << endl);
			return true;
		}

		template<class Voxel>
		bool TemplateVoxelMap<Voxel>::merge(const GpuVoxelsMapSharedPtr other, const Vector3f& metric_offset, const BitVoxelMeaning* new_meaning)
		{
			// scale to voxel size
			const Vector3i tmp = (metric_offset / m_voxel_side_length).template cast<int32_t>();
			return this->merge(other, tmp, new_meaning);
		}

		template<class Voxel>
		bool TemplateVoxelMap<Voxel>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning) {
			if (voxel_offset != Vector3i::Zero()) 
			{
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, "You tried to apply an offset while merging a VoxelList into a DistanceVoxelMap (not supported yet)!" << endl);
				return false;
			}

			std::scoped_lock lock(this->m_mutex, other->m_mutex);

			//TODO:: reflection for map_type to avoid switch case with static compilation
			switch (other->getMapType())
			{
			case MT_BITVECTOR_VOXELLIST:
			{
				auto* voxelList = other->as<voxellist::TemplateVoxelList<BitVectorVoxel, MapVoxelID>>();
				insertCoordinateList(voxelList->getDeviceCoords(), eBVM_OCCUPIED);
				return true;
			}

			case MT_PROBAB_VOXELLIST:
			{
				auto* voxelList = other->as<voxellist::TemplateVoxelList<ProbabilisticVoxel, MapVoxelID>>();
				insertCoordinateList(voxelList->getDeviceCoords(), eBVM_OCCUPIED);
				return true;
			}

			case MT_COUNTING_VOXELLIST:
			{
				auto* voxelList = other->as<voxellist::TemplateVoxelList<CountingVoxel, MapVoxelID>>();
				insertCoordinateList(voxelList->getDeviceCoords(), eBVM_OCCUPIED);
				return true;
			}

			default:
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
				return false;
			}
		}

		template<class Voxel>
		Vector3ui TemplateVoxelMap<Voxel>::getDimensions() const
		{
			return m_dim;
		}

		template<class Voxel>
		Vector3f TemplateVoxelMap<Voxel>::getMetricDimensions() const
		{
			return  m_dim.template cast<float>() * getVoxelSideLength();
		}

		template<class Voxel>
		void TemplateVoxelMap<Voxel>::clone(const TemplateVoxelMap<Voxel>& other)
		{
			if (this->m_voxel_side_length != other.m_voxel_side_length || this->m_dim != other.m_dim)
			{
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, "Voxelmap cannot be cloned since map reference dimensions or voxel side length are not equal" << endl);
				return;
			}
			
			std::scoped_lock lock(this->m_mutex, other.m_mutex);

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			thrust::copy(other.m_dev_data.begin(), other.m_dev_data.end(), this->m_dev_data.begin());
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
		}

		// ------ END Global API functions ------


#ifdef ALTERNATIVE_CHECK
#undef ALTERNATIVE_CHECK
#endif



	}// end of namespace voxelmap
} // end of namespace gpu_voxels

#endif