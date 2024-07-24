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
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_DATACONTEXT_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_DATACONTEXT_H_INCLUDED

#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <glm/glm.hpp>

#include <gpu_visualization/Primitive.h>

namespace gpu_voxels {
	namespace visualization {

		typedef std::pair<glm::vec4, glm::vec4> colorPair;
		class DataContext
		{
		public:

			DataContext();
			virtual ~DataContext() = default;

			void set_color(const colorPair& pair, size_t i);

			void updateTotalNumVoxels();

			virtual void updateVBOOffsets();

			virtual void updateCudaLaunchVariables(Vector3ui supervoxel_size = Vector3ui::Ones());

			/**
			 * Returns the number of vertices of the specified type in the current view.
			 * If i is not a valid type 0 will be returned.
			 */
			[[nodiscard]] uint32_t getNumberOfVerticesOfType(uint32_t i) const;

			/**
			 * Returns the number of vertices in the current view.
			 */
			[[nodiscard]] uint32_t getNumberOfVertices() const;

			/**
			 * Returns the size of all vertices in the current view in byte.
			 */
			[[nodiscard]] uint32_t getSizeOfVertices() const;

			[[nodiscard]] size_t getSizeForBuffer() const;

			[[nodiscard]] uint32_t getOffset(uint32_t i) const;

			[[nodiscard]] dim3 num_blocks() const;

			[[nodiscard]] dim3 threads_per_block() const;

			[[nodiscard]] Probability occupancy_threshold() const;

			void set_occupancy_threshold(Probability threshold, const std::string& name = "N/A");

			[[nodiscard]] const std::string& map_name() const;

			void set_num_voxels_per_type(const thrust::device_vector<uint32_t>& num_voxels_per_type);

			[[nodiscard]] size_t voxel_types() const;

			cudaGraphicsResource** cuda_ressource();


			thrust::device_vector<uint32_t> m_d_vbo_segment_voxel_capacities;
			thrust::device_vector<uint32_t> m_d_vbo_offsets;

		public:

			// the name of the data structure
			std::string m_map_name;

			//determines if the data context should be drawn
			bool m_draw_context;

			// contains the colors for each type
			thrust::host_vector<colorPair> m_colors;
			// the OpenGL buffer for this data structure
			GLuint m_vbo;
			// indicates if the vbo may be drawn right now
			bool m_vbo_draw_able;
			// the current size of the VBO
			size_t m_cur_vbo_size;
			// the maximum size of the vbo <=> 0 is no limit
			size_t m_max_vbo_size;
			// the cudaGraphicsResource for this data structure
			cudaGraphicsResource* m_cuda_ressources;
			// the default primitive of this context
			std::unique_ptr<Primitive> m_default_prim;

			// the minimum occupancy probability for the context
			Probability m_occupancy_threshold;

			// an offset for the data structure
			glm::vec3 m_translation_offset;

			// total number of occupied voxels in the current view <=> sum(num_voxels_per_type)
			uint32_t m_total_num_voxels;
			//number of occupied voxels of each type
			thrust::host_vector<uint32_t> m_num_voxels_per_type;

			//the vbo segment sizes
			thrust::host_vector<uint32_t> m_vbo_segment_voxel_capacities;

			thrust::host_vector<uint32_t> m_vbo_offsets;

			// mapping from type to segment
			thrust::host_vector<uint8_t> m_types_segment_mapping;
			bool m_has_draw_type_flipped;

			//cuda kernel launch variable
			dim3 m_threads_per_block;
			dim3 m_num_blocks;

			//thrust::device_vector<uint32_t> m_d_num_voxels_per_type;
		};
	} // end of namespace visualization
} // end of namespace gpu_voxels
#endif