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
 * \author  Matthias Wagner
 * \date    2014-02-10
 *
 *\brief   Saves all necessary stuff to draw a voxel map.
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_VOXEL_MAP_CONTEXT_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_VOXEL_MAP_CONTEXT_H_INCLUDED

#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

//#include <gpu_voxels/voxelmap/VoxelMap.h>
#include "Cuboid.h"
#include "Sphere.h"
#include "DataContext.h"
#include "TemplatedVoxelMapVisual.hpp"

namespace gpu_voxels {
	namespace visualization {

		typedef std::pair<glm::vec4, glm::vec4> colorPair;

		class VoxelmapContext : public DataContext
		{
		public:

			VoxelmapContext()
				: m_voxelMap(nullptr)
			{
			}

			/**
			 * Create a default context for the given voxel map.
			 */
			VoxelmapContext(voxelmap::AbstractVoxelMapVisual* map, std::string map_name)
				: m_voxelMap(map)
			{
				m_map_name = std::move(map_name);

				updateCudaLaunchVariables();
				m_default_prim = std::make_unique<Cuboid>(glm::vec4(0.f, 0.f, 0.f, 1.f),
					glm::vec3(0.f, 0.f, 0.f),
					glm::vec3(1.f, 1.f, 1.f));
			}

			void updateVBOOffsets() override
			{
				thrust::exclusive_scan(m_vbo_segment_voxel_capacities.begin(),
					m_vbo_segment_voxel_capacities.end(),
					m_vbo_offsets.begin());
				m_d_vbo_offsets = m_vbo_offsets;
			}

			void updateCudaLaunchVariables(Vector3ui supervoxel_size = Vector3ui::Ones()) override
			{
				auto map = m_voxelMap;
				m_threads_per_block = dim3(8, 8, 8);
				const Vector3ui thread_reduce_factor = supervoxel_size.cwiseMax(Vector3ui(6, 6, 6));
				
				m_num_blocks = dim3(
					static_cast<unsigned>(ceil(static_cast<float>(map->getDimensions().x()) / (m_threads_per_block.x * thread_reduce_factor.x()))),
					static_cast<unsigned>(ceil(static_cast<float>(map->getDimensions().y()) / (m_threads_per_block.y * thread_reduce_factor.y()))),
						static_cast<unsigned>(ceil(static_cast<float>(map->getDimensions().z()) / (m_threads_per_block.z * thread_reduce_factor.z()))));
			}

			// the voxel map of this context
			voxelmap::AbstractVoxelMapVisual* m_voxelMap;
		};

	}  // end of ns
}  // end of ns

#endif
