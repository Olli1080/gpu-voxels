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
 * \author  Florian Drews
 * \author  Christian Juelg
 * \date    2012-06-22
 *
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_CUDA_DATATYPES_H_INCLUDED
#define GPU_VOXELS_CUDA_DATATYPES_H_INCLUDED

#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_vectors.hpp>
#include <gpu_voxels/helpers/cuda_matrices.hpp>
#include <thrust/host_vector.h>

#include "BitVector.h"
#include "common_defines.h"

#define BALLOT(PREDICATE) __ballot_sync(__activemask(), PREDICATE)

namespace gpu_voxels
{
	struct MetaPointCloudStruct
	{
		uint16_t num_clouds;
		uint32_t accumulated_cloud_size;
		uint32_t* cloud_sizes;
		Vector3f** clouds_base_addresses;

		__device__ __host__
			MetaPointCloudStruct()
			: num_clouds(0), accumulated_cloud_size(0),
			cloud_sizes(nullptr),
			clouds_base_addresses(nullptr)
		{}
	};

	struct MetaPointCloudStructLocal
	{
		uint16_t num_clouds;
		uint32_t accumulated_cloud_size;
		thrust::host_vector<uint32_t> cloud_sizes;
		std::vector<std::vector<Vector3f>::iterator> clouds_base_addresses;

		__device__ __host__
			MetaPointCloudStructLocal()
			: num_clouds(0), accumulated_cloud_size(0)
		{}
	};

	struct Cube
	{
		// Default constructor is needed
		__device__ __host__ Cube()
		{
			m_side_length = 0;
			m_position = Vector3ui(0);
			m_type_vector.setBit(eBVM_UNDEFINED);
		}

		__device__ __host__ Cube(uint32_t side_length, Vector3ui position, BitVoxelMeaning type)
		{
			m_side_length = side_length;
			m_position = position;
			m_type_vector.setBit(type);
		}

		__device__ __host__ Cube(uint32_t side_length, Vector3ui position, BitVector<visualization::MAX_DRAW_TYPES> type_vector)
		{
			m_side_length = side_length;
			m_position = position;
			m_type_vector = type_vector;
		}

		// the length of the edges
		uint32_t m_side_length;
		// the position of the lower left front corner of the cube
		Vector3ui m_position;

		// type bitvector
		BitVector<visualization::MAX_DRAW_TYPES> m_type_vector;

	};
} // end of namespace
#endif