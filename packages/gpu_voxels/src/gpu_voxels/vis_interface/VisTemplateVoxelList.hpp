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
* \date    2015-06-10
*
*/
//----------------------------------------------------------------------
#pragma once

#include "VisTemplateVoxelList.h"

#include <thrust/device_vector.h>

namespace gpu_voxels {

	template <class Voxel, typename VoxelIDType>
	VisTemplateVoxelList<Voxel, VoxelIDType>::VisTemplateVoxelList(voxellist::TemplateVoxelList<Voxel, VoxelIDType>* voxellist, std::string map_name)
		: VisProvider(shm_segment_name_voxellists, map_name),
		m_voxellist(voxellist),
		m_shm_memHandle(nullptr), /**/
		m_dev_buffer_1(nullptr),
		m_dev_buffer_2(nullptr),
		m_shm_bufferSwapped(nullptr),
		m_shm_num_cubes(nullptr),
		m_internal_buffer_1(false),
		m_shm_voxellist_type(nullptr) /**/
	{
	}

	template <class Voxel, typename VoxelIDType>
	VisTemplateVoxelList<Voxel, VoxelIDType>::~VisTemplateVoxelList() = default;

	template <class Voxel, typename VoxelIDType>
	bool VisTemplateVoxelList<Voxel, VoxelIDType>::visualize(const bool force_repaint)
	{
		openOrCreateSegment();
		if (m_shm_memHandle == nullptr)
		{
			uint32_t shared_mem_id;
			// there should only be one segment of number_of_voxelmaps
			std::pair<uint32_t*, std::size_t> r = m_segment.find<uint32_t>(
				shm_variable_name_number_of_voxellists.c_str());
			if (r.second == 0)
			{ // if it doesn't exists ..
				m_segment.construct<uint32_t>(shm_variable_name_number_of_voxellists.c_str())(1);
				shared_mem_id = 0;
			}
			else
			{ // if it exists increase it by one
				shared_mem_id = *r.first;
				(*r.first)++;
			}
			// get shared memory pointer
			std::string id = std::to_string(shared_mem_id);
			m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(std::string(shm_variable_name_voxellist_handler_dev_pointer + id).c_str())(cudaIpcMemHandle_t());
			m_shm_num_cubes = m_segment.find_or_construct<uint32_t>(std::string(shm_variable_name_voxellist_num_voxels + id).c_str())(static_cast<uint32_t>(0));
			m_shm_bufferSwapped = m_segment.find_or_construct<bool>(std::string(shm_variable_name_voxellist_buffer_swapped + id).c_str())(false);
			std::cout << "Name of shared buffer swapped: " << std::string(shm_variable_name_voxellist_buffer_swapped + id).c_str() << "." << std::endl;
			m_shm_mapName = m_segment.find_or_construct_it<char>(std::string(shm_variable_name_voxellist_name + id).c_str())[m_map_name.size()](m_map_name.data());
			m_shm_voxellist_type = m_segment.find_or_construct<MapType>(std::string(shm_variable_name_voxellist_type + id).c_str())(m_voxellist->getMapType());

		}

		if (*m_shm_bufferSwapped == false && force_repaint)
		{
			uint32_t cube_buffer_size;
			Cube* d_cubes_buffer;

			if (m_internal_buffer_1)
			{
				// extractCubes() allocates memory for the m_dev_buffer_1, if the pointer is nullptr
				m_voxellist->extractCubes(&m_dev_buffer_1);
				cube_buffer_size = m_dev_buffer_1->size();
				d_cubes_buffer = thrust::raw_pointer_cast(m_dev_buffer_1->data());
				m_internal_buffer_1 = false;
			}
			else
			{
				// extractCubes() allocates memory for the m_dev_buffer_2, if the pointer is nullptr
				m_voxellist->extractCubes(&m_dev_buffer_2);
				cube_buffer_size = m_dev_buffer_2->size();
				d_cubes_buffer = thrust::raw_pointer_cast(m_dev_buffer_2->data());
				m_internal_buffer_1 = true;
			}

			if (cube_buffer_size > 0)
			{
				HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, d_cubes_buffer));
				*m_shm_num_cubes = cube_buffer_size;
				*m_shm_bufferSwapped = true;
			}
			else {
				*m_shm_memHandle = cudaIpcMemHandle_t(); // an empty memory handle, since no memory is allocated in this case
				*m_shm_num_cubes = 0;
				*m_shm_bufferSwapped = true;
			}
			return true;
		}
		return false;
	}

	template <class Voxel, typename VoxelIDType>
	uint32_t VisTemplateVoxelList<Voxel, VoxelIDType>::getResolutionLevel()
	{
		return 0; // todo query correct resolution from visualizer like VisNTree
	}

} // namespace gpu_voxels