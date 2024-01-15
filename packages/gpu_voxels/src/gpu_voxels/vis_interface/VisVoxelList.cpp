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

#include "VisVoxelList.h"

namespace gpu_voxels {

	VisVoxelList::VisVoxelList(voxellist::AbstractVoxelList* voxellist, std::string map_name)
		: VisProvider(shm_segment_name_voxellists, map_name),
		m_voxellist(voxellist),
		m_shm_memHandle(nullptr), /**/
		m_shm_list_size(nullptr), /**/
		m_shm_VoxelSize(nullptr), /**/
		m_shm_voxellist_type(nullptr), /**/
		m_shm_voxellist_changed(nullptr)
	{
	}

	VisVoxelList::~VisVoxelList() = default;

	bool VisVoxelList::visualize(const bool force_repaint)
	{
		if (!force_repaint)
			return false;

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
			m_shm_list_size = m_segment.find_or_construct<uint32_t>(std::string(shm_variable_name_voxellist_num_voxels + id).c_str())(static_cast<uint32_t>(0));
			m_shm_VoxelSize = m_segment.find_or_construct<float>(std::string(shm_variable_name_voxel_side_length + id).c_str())(0.0f);
			m_shm_mapName = m_segment.find_or_construct_it<char>(std::string(shm_variable_name_voxellist_name + id).c_str())[m_map_name.size()](m_map_name.data());
			m_shm_voxellist_type = m_segment.find_or_construct<MapType>(std::string(shm_variable_name_voxellist_type + id).c_str())(m_voxellist->getMapType());

			//      m_shm_voxellist_changed = m_segment.find_or_construct<bool>(
			//          std::string(shm_variable_name_voxellist_buffer_swapped + id.str()).c_str())(true);

		}
		// first open or create and the set the values
		HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, m_voxellist->getVoidDeviceDataPtr()));
		*m_shm_list_size = m_voxellist->getDimensions().x();
		*m_shm_VoxelSize = m_voxellist->getVoxelSideLength();
		*m_shm_voxellist_changed = true;

		return true;
	}

	uint32_t VisVoxelList::getResolutionLevel()
	{
		return 0; // todo query correct resolution from visualizer like VisNTree
	}

}