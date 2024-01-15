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
#ifndef GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_H_INCLUDED_VISUAL
#define GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_H_INCLUDED_VISUAL

//#include <vector>

//#include <cuda_runtime.h>

//#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>

#include <gpu_voxels/helpers/MathHelpers.h>

#include <gpu_voxels/voxel/DistanceVoxel.hpp>
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/voxel/ProbabilisticVoxel.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Data structure and according operations
 */
namespace gpu_voxels {
	namespace voxelmap {
		
		class GpuVoxelsMapVisual;
		typedef std::shared_ptr<GpuVoxelsMapVisual> GpuVoxelsMapVisualSharedPtr;

		class GpuVoxelsMapVisual
		{
		public:
			//! Constructor
			GpuVoxelsMapVisual() = default;

			//! Destructor
			virtual ~GpuVoxelsMapVisual() = default;

			//to ensure these methods can't be called
			GpuVoxelsMapVisual(const GpuVoxelsMapVisual&) = delete;
			const GpuVoxelsMapVisual& operator=(const GpuVoxelsMapVisual&) = delete;

			/*!
			 * \brief getMapType returns the type of the map
			 * \return the type of the map
			 */
			MapType getMapType() const;

			/*!
			 * \brief getMemoryUsage
			 * \return Returns the size of the used memory in byte
			 */
			virtual std::size_t getMemoryUsage() const = 0;

			/*!
			 * \brief getDimensions
			 * \return Returns the dimensions of the map in voxels.
			 */
			virtual Vector3ui getDimensions() const = 0;

			/*!
			 * \brief getMetricDimensions
			 * \return Returns the dimensions of the map in meter.
			 */
			virtual Vector3f getMetricDimensions() const = 0;

			// public to allow lock_guard(OtherMapTypePtr->m_mutex)
			//mutable std::recursive_mutex m_mutex;

		protected:

			MapType m_map_type;
		};

		class AbstractVoxelMapVisual : public GpuVoxelsMapVisual
		{
		public:

			~AbstractVoxelMapVisual() override = default;

			//! get pointer to data array on device
			virtual void* getVoidDeviceDataPtr() = 0;

			virtual const void* getConstVoidDeviceDataPtr() const = 0;

			//! get the side length of the voxels.
			virtual float getVoxelSideLength() const = 0;

			//! get the number of bytes that is required for the voxelmap
			size_t getMemoryUsage() const override = 0;

			virtual MapType getTemplateType() const = 0;
		};
		
		template<class Voxel>
		class TemplateVoxelMapVisual : public AbstractVoxelMapVisual
		{
		public:

			/*!
			 * This constructor does NOT create a new voxel map on the GPU.
			 * The new object will represent the voxel map specified in /p dev_data.
			 * Warning: Not all member variables will be set correctly for the map.
			 */
			TemplateVoxelMapVisual(Voxel* dev_data, const Vector3ui& dim, float voxel_side_length, MapType map_type)
				: m_dim(dim), m_limits(dim.template cast<float>() * voxel_side_length),
				m_voxel_side_length(voxel_side_length), m_dev_data(dev_data) //TODO:: here is a bug!!
			{
				this->m_map_type = map_type;
				computeLinearLoad(getVoxelMapSize(), m_blocks, m_threads);
			}

			//! Destructor
			~TemplateVoxelMapVisual() override = default;

			/* ======== getter functions ======== */

			//! get pointer to data array on device
			Voxel* getDeviceDataPtr()
			{
				//return m_dev_data.data().get();
				return m_dev_data.get();
			}

			const Voxel* getConstDeviceDataPtr() const
			{
				//return m_dev_data.data().get();
				return m_dev_data.get();
			}

			/*const thrust::device_vector<Voxel>& getDeviceData() const
			{
				//return m_dev_data;
				return {};
			}*/

			void* getVoidDeviceDataPtr() override
			{
				//return thrust::raw_pointer_cast(m_dev_data.data());
				return m_dev_data.get();
			}

			const void* getConstVoidDeviceDataPtr() const override
			{
				//return thrust::raw_pointer_cast(m_dev_data.data());
				return m_dev_data.get();
			}

			//! get the number of voxels held in the voxelmap
			uint32_t getVoxelMapSize() const
			{
				return m_dim.sum();
			}

			//! get the side length of the voxels.
			float getVoxelSideLength() const override
			{
				return m_voxel_side_length;
			}

			//virtual void gatherVoxelsByIndex(thrust::device_ptr<unsigned int> dev_indices_begin, thrust::device_ptr<unsigned int> dev_indices_end, thrust::device_ptr<Voxel> dev_output_begin) {};

			std::size_t getMemoryUsage() const override
			{
				return m_dim.sum() * sizeof(Voxel);
			}

			Vector3ui getDimensions() const override { return m_dim; }

			Vector3f getMetricDimensions() const override { return m_dim.template cast<float>() * getVoxelSideLength(); }

			// ------ END Global API functions ------


		protected:

			/* ======== Variables with content on host ======== */
			const Vector3ui m_dim;
			const Vector3f m_limits;
			float m_voxel_side_length;

			uint32_t m_blocks;
			uint32_t m_threads;

			/* ======== Variables with content on device ======== */

			/*! VoxelMap data on device.
			 *  storage format is: index = z * dim.x * dim.y + y * dim.x + x  */

			thrust::device_ptr<Voxel> m_dev_data;
		};


		template<std::size_t length>
		class BitVoxelMapVisual : public TemplateVoxelMapVisual<BitVoxel<length>>
		{
		public:
			typedef BitVoxel<length> Voxel;
			typedef TemplateVoxelMapVisual<Voxel> Base;
			
			BitVoxelMapVisual(Voxel* dev_data, const Vector3ui& dim, float voxel_side_length, MapType map_type)
				: Base(dev_data, dim, voxel_side_length, map_type)
			{}

			~BitVoxelMapVisual() override = default;

			[[nodiscard]] MapType getTemplateType() const override { return MT_BITVECTOR_VOXELMAP; }
		};

		class ProbVoxelMapVisual : public TemplateVoxelMapVisual<ProbabilisticVoxel>
		{
		public:

			typedef ProbabilisticVoxel Voxel;
			typedef TemplateVoxelMapVisual<Voxel> Base;
			
			ProbVoxelMapVisual(Voxel* dev_data, const Vector3ui& dim, float voxel_side_length, MapType map_type);
			~ProbVoxelMapVisual() override = default;

			MapType getTemplateType() const override { return MT_PROBAB_VOXELMAP; }
		};


		typedef unsigned int uint;
		
		typedef gpu_voxels::DistanceVoxel::extract_byte_distance::free_space_t free_space_t;
		typedef gpu_voxels::DistanceVoxel::init_floodfill_distance::manhattan_dist_t manhattan_dist_t;

		class DistanceVoxelMapVisual : public TemplateVoxelMapVisual<DistanceVoxel>
		{
		public:

			typedef DistanceVoxel Voxel;
			typedef TemplateVoxelMapVisual<Voxel> Base;
			
			DistanceVoxelMapVisual(Voxel* dev_data, const Vector3ui& dim, float voxel_side_length, MapType map_type);
			~DistanceVoxelMapVisual() override = default;

			MapType getTemplateType() const override { return MT_DISTANCE_VOXELMAP; }
		};

		typedef BitVoxelMapVisual<BIT_VECTOR_LENGTH> BitVectorVoxelMapVisual;
	} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif