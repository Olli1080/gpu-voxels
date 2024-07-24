#include "DataContext.h"

#include <numeric>

#include <gpu_visualization/logging/logging_visualization.h>

namespace gpu_voxels
{
	namespace visualization
	{
		DataContext::DataContext() :
			m_draw_context(true), m_vbo(), m_vbo_draw_able(false), m_cur_vbo_size(1), m_max_vbo_size(0),
			m_cuda_ressources(), m_occupancy_threshold(0), m_translation_offset(0.f), m_total_num_voxels(0)
		{
			m_threads_per_block = dim3(10, 10, 10);
			m_num_blocks = dim3(1, 1, 1);

			m_default_prim = nullptr;

			//insert some default colors
			// The ordering has to be the same as for BitVoxelMeaning enum
			// defined in gpu_voxels/helpers/common_defines.h
			colorPair p;
			p.first = p.second = glm::vec4(1.f, 1.f, 0.f, 1.f);
			m_colors.push_back(p);/*color for voxel type eBVM_FREE yellow*/

			p.first = p.second = glm::vec4(0.f, 1.f, 0.f, 1.f);
			m_colors.push_back(p);/*color for voxel type eBVM_OCCUPIED green*/

			p.first = p.second = glm::vec4(1.f, 0.f, 0.f, 1.f);
			m_colors.push_back(p);/*color for voxel type eBVM_COLLISION red*/

			p.first = p.second = glm::vec4(1.f, 0.f, 1.f, 1.f);
			m_colors.push_back(p);/*color for voxel type eBVM_UNKNOWN magenta*/


			// swept volume colors blend altering fashion from yellow to blue,
			// and from
			float increment = 1.f / static_cast<float>(eBVM_SWEPT_VOLUME_END - eBVM_SWEPT_VOLUME_START);
			float change = 0.f;
			size_t step = 0;
			for (size_t i = eBVM_SWEPT_VOLUME_START; i <= eBVM_SWEPT_VOLUME_END; ++i)
			{
				change = step * increment;

				p.first = p.second = glm::vec4(1.f - change, (step % 2)
					? 1.f // yellow to light green
					: 0.f, // red to blue
					change, 1.f);

				++step;
				m_colors.push_back(p);
			}

			p.first = p.second = glm::vec4(0.f, 1.f, 1.f, 1.f);
			m_colors.push_back(p);/*color for voxel type eBVM_UNDEFINED(255) cyan*/

			m_num_voxels_per_type.resize(MAX_DRAW_TYPES);
			//m_d_num_voxels_per_type = m_num_voxels_per_type;

			m_vbo_segment_voxel_capacities.resize(MAX_DRAW_TYPES);
			m_d_vbo_segment_voxel_capacities = m_vbo_segment_voxel_capacities;

			m_vbo_offsets.resize(MAX_DRAW_TYPES);
			m_d_vbo_offsets = m_vbo_offsets;

			m_types_segment_mapping = thrust::host_vector<uint8_t>(MAX_DRAW_TYPES, 0);
			m_has_draw_type_flipped = true;
		}

		void DataContext::set_color(const colorPair& pair, size_t i)
		{
			m_colors[i] = pair;
		}

		void DataContext::updateTotalNumVoxels()
		{
			m_total_num_voxels = std::accumulate(m_num_voxels_per_type.begin(), m_num_voxels_per_type.end(), 0);
		}

		void DataContext::updateVBOOffsets()
		{
		}

		void DataContext::updateCudaLaunchVariables(Vector3ui supervoxel_size)
		{
		}

		uint32_t DataContext::getNumberOfVerticesOfType(uint32_t i) const
		{
			return i < m_num_voxels_per_type.size() ? m_num_voxels_per_type[i] * 36 : 0;
		}

		uint32_t DataContext::getNumberOfVertices() const
		{
			return m_total_num_voxels * 36;
		}

		uint32_t DataContext::getSizeOfVertices() const
		{
			return m_total_num_voxels * 36/*vertices per voxel*/ * 3 * sizeof(float)/*size of a vertex*/;
		}

		size_t DataContext::getSizeForBuffer() const
		{
			return m_total_num_voxels * SIZE_OF_TRANSLATION_VECTOR;
		}

		uint32_t DataContext::getOffset(uint32_t i) const
		{
			return i < m_vbo_offsets.size() ? m_vbo_offsets[i] : 0;
		}

		dim3 DataContext::num_blocks() const
		{
			return m_num_blocks;
		}

		dim3 DataContext::threads_per_block() const
		{
			return m_num_blocks;
		}

		Probability DataContext::occupancy_threshold() const
		{
			return m_occupancy_threshold;
		}

		void DataContext::set_occupancy_threshold(Probability threshold, const std::string& name)
		{
			if (threshold > MAX_PROBABILITY)
			{
				LOGGING_WARNING_C(
					Visualization,
					DataContext, //TODO:: use actual values for max_prob* and min_*
					"Occupancy_threshold of " << name << " is too big (" << threshold << "). MAX_PROBABILITY (127) is used instead." << endl);
				m_occupancy_threshold = MAX_PROBABILITY;
			}
			else if (threshold < MIN_PROBABILITY)
			{
				LOGGING_WARNING_C(
					Visualization,
					DataContext,
					"Occupancy_threshold of " << name << " is too small (" << threshold << "). MIN_PROBABILITY (-127) is used instead." << endl);
				m_occupancy_threshold = MIN_PROBABILITY;
			}
			else if (threshold == 0)
			{
				LOGGING_WARNING_C(Visualization, DataContext,
					"Occupancy_threshold of " << name << " is zero." << endl);
				m_occupancy_threshold = 0;
			}
			else
			{
				m_occupancy_threshold = threshold;
			}
		}

		const std::string& DataContext::map_name() const
		{
			return m_map_name;
		}

		void DataContext::set_num_voxels_per_type(const thrust::device_vector<uint32_t>& num_voxels_per_type)
		{
			m_num_voxels_per_type = num_voxels_per_type;
			updateVBOOffsets();
			updateTotalNumVoxels();
		}

		size_t DataContext::voxel_types() const
		{
			return m_num_voxels_per_type.size();
		}

		cudaGraphicsResource** DataContext::cuda_ressource()
		{
			return &m_cuda_ressources;
		}
	}
}