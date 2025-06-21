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
 * \date    2014-06-18
 *
 */
 //----------------------------------------------------------------------/*
#ifndef VISNTREE_CPP_
#define VISNTREE_CPP_

#include <gpu_voxels/octree/VisNTree.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <thrust/device_vector.h>

namespace gpu_voxels {
	namespace NTree {

		template<typename InnerNode, typename LeafNode>
		VisNTree<InnerNode, LeafNode>::VisNTree(MyNTree* ntree, std::string map_name) :
			VisProvider(shm_segment_name_octrees, map_name), m_ntree(ntree), m_shm_memHandle(nullptr), m_min_level(
				UINT_MAX), m_shm_superVoxelSize(nullptr), m_shm_numCubes(nullptr), m_shm_bufferSwapped(nullptr), m_internal_buffer_1(false),
			m_d_cubes_1(nullptr), m_d_cubes_2(nullptr)
		{}

		template<typename InnerNode, typename LeafNode>
		VisNTree<InnerNode, LeafNode>::~VisNTree() = default;

		template<typename InnerNode, typename LeafNode>
		bool VisNTree<InnerNode, LeafNode>::visualize(const bool force_repaint)
		{
			openOrCreateSegment();
			if (m_shm_memHandle == nullptr) // do this only once
			{
				uint32_t shared_mem_id;
				// there should only be one segment of number_of_octrees
				std::pair<uint32_t*, std::size_t> r = m_segment.find<uint32_t>(shm_variable_name_number_of_octrees.c_str());
				if (r.second == 0)
				{ // if it doesn't exist ..
					m_segment.construct<uint32_t>(shm_variable_name_number_of_octrees.c_str())(1);
					shared_mem_id = 0;
				}
				else
				{ // if it exit increase it by one
					shared_mem_id = *r.first;
					(*r.first)++;
				}

				// get shared memory pointer
				const std::string id = std::to_string(shared_mem_id);
				m_shm_superVoxelSize = m_segment.find_or_construct<uint32_t>(shm_variable_name_super_voxel_size.c_str())(1);
				m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(std::string(shm_variable_name_octree_handler_dev_pointer + id).c_str())(cudaIpcMemHandle_t());
				m_shm_numCubes = m_segment.find_or_construct<uint32_t>(std::string(shm_variable_name_number_cubes + id).c_str())(0);
				m_shm_bufferSwapped = m_segment.find_or_construct<bool>(std::string(shm_variable_name_octree_buffer_swapped + id).c_str())(false);
				m_shm_mapName = m_segment.find_or_construct_it<char>(std::string(shm_variable_name_octree_name + id).c_str())[m_map_name.size()](m_map_name.data());

			}

			const uint32_t tmp = *m_shm_superVoxelSize - 1;
			// m_shm_bufferSwapped tells, if visualizer already rendered the frame
			// m_internal_buffer tells, which buffer should be used
			if (*m_shm_bufferSwapped == false && (tmp != m_min_level || force_repaint))
			{
				m_min_level = tmp;

				uint32_t cube_buffer_size;
				Cube* d_cubes_buffer;

				if (m_internal_buffer_1)
				{
					// extractCubes() allocates memory for the d_cubes_1, if the pointer is nullptr
					cube_buffer_size = m_ntree->extractCubes(m_d_cubes_1, nullptr, m_min_level);
					d_cubes_buffer = thrust::raw_pointer_cast(m_d_cubes_1->data());
					m_internal_buffer_1 = false;
				}
				else {
					// extractCubes() allocates memory for the d_cubes_2, if the pointer is nullptr
					cube_buffer_size = m_ntree->extractCubes(m_d_cubes_2, nullptr, m_min_level);
					d_cubes_buffer = thrust::raw_pointer_cast(m_d_cubes_2->data());
					m_internal_buffer_1 = true;
				}

				HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, d_cubes_buffer));
				*m_shm_numCubes = cube_buffer_size;
				*m_shm_bufferSwapped = true;

				return true;
			}
			return false;
		}

		template<typename InnerNode, typename LeafNode>
		uint32_t VisNTree<InnerNode, LeafNode>::getResolutionLevel()
		{
			if (m_shm_superVoxelSize != nullptr)
				return *m_shm_superVoxelSize - 1;

			return 0;
		}

	} // end of ns
} // end of ns

#endif