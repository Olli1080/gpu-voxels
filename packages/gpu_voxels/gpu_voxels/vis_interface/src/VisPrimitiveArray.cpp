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
 * \author  Andreas Hermann
 * \date    2014-12-15
 *
 */
 //----------------------------------------------------------------------/*
#include <gpu_voxels/vis_interface/VisPrimitiveArray.h>
#include <gpu_voxels/helpers/cuda_handling.h>
#include <cstdio>

std::ostream& operator << (std::ostream& os, const cudaIpcMemHandle_t& handle)
{
	os << static_cast<int>(handle.reserved[0]);
	for (int i = 1; i < CUDA_IPC_HANDLE_SIZE; ++i)
		os << " " << static_cast<int>(handle.reserved[i]);

	return os;
}

namespace gpu_voxels {

	VisPrimitiveArray::VisPrimitiveArray(primitive_array::PrimitiveArray* primitive_array, std::string array_name) :
		VisProvider(shm_segment_name_primitive_array, array_name), /**/
		m_primitive_array(primitive_array), /**/
		m_shm_memHandle(nullptr), /**/
		m_shm_primitive_diameter(nullptr), /**/
		m_shm_num_primitives(nullptr), /**/
		m_shm_primitive_type(nullptr), /**/
		m_shm_primitive_array_changed(nullptr)
	{
	}

	VisPrimitiveArray::~VisPrimitiveArray() = default;

	bool VisPrimitiveArray::visualize(const bool force_repaint)
	{
		if (!force_repaint)
			return false;

		openOrCreateSegment();
		if (m_shm_memHandle == nullptr)
		{
			uint32_t shared_mem_id;
			// there should only be one segment of number_of_primitive_arrays
			std::pair<uint32_t*, std::size_t> r = m_segment.find<uint32_t>(
				shm_variable_name_number_of_primitive_arrays.c_str());
			if (r.second == 0)
			{ // if it doesn't exists ..
				m_segment.construct<uint32_t>(shm_variable_name_number_of_primitive_arrays.c_str())(1);
				shared_mem_id = 0;
			}
			else
			{ // if it exists increase it by one
				shared_mem_id = *r.first;
				(*r.first)++;
			}
			// get shared memory pointer
			std::string id = std::to_string(shared_mem_id);
			m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(std::string(shm_variable_name_primitive_array_handler_dev_pointer + id).c_str())(cudaIpcMemHandle_t());
			m_shm_primitive_diameter = m_segment.find_or_construct<float>(std::string(shm_variable_name_primitive_array_prim_diameter + id).c_str())(0.0f);
			m_shm_mapName = m_segment.find_or_construct_it<char>(std::string(shm_variable_name_primitive_array_name + id).c_str())[m_map_name.size()](m_map_name.data());
			m_shm_primitive_type = m_segment.find_or_construct<primitive_array::PrimitiveType>(std::string(shm_variable_name_primitive_array_type + id).c_str())(m_primitive_array->getPrimitiveType());
			m_shm_num_primitives = m_segment.find_or_construct<uint32_t>(std::string(shm_variable_name_primitive_array_number_of_primitives + id).c_str())(m_primitive_array->getNumPrimitives());
			m_shm_primitive_array_changed = m_segment.find_or_construct<bool>(std::string(shm_variable_name_primitive_array_data_changed + id).c_str())(true);
		}
		// first open or create and then set the values
		// but only, if data is available
		if (m_primitive_array->getVoidDeviceDataPtr())
		{
			HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, m_primitive_array->getVoidDeviceDataPtr()));
			*m_shm_primitive_diameter = m_primitive_array->getDiameter();
			*m_shm_primitive_type = m_primitive_array->getPrimitiveType();
			*m_shm_num_primitives = m_primitive_array->getNumPrimitives();
			*m_shm_primitive_array_changed = true;
		}
		else {
			*m_shm_primitive_array_changed = false;
		}

		//    // wait till data was read by visualizer. Otherwise a
		//    while(*m_shm_voxelmap_changed)
		//      usleep(10000); // sleep 10 ms
		return true;
	}

} // end of ns