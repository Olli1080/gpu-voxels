#include "CubelistContext.h"

#include <gpu_visualization/Cuboid.h>

namespace gpu_voxels {
	namespace visualization {

		CubelistContext::CubelistContext(const std::string& map_name)
			: m_d_cubes(nullptr),
			m_number_of_cubes(0)
		{
			m_map_name = map_name;
			m_default_prim = std::make_unique<Cuboid>(glm::vec4(0.f, 0.f, 0.f, 1.f),
				glm::vec3(0.f, 0.f, 0.f),
				glm::vec3(1.f, 1.f, 1.f));
			m_threads_per_block = dim3(cMAX_THREADS_PER_BLOCK);
			m_num_blocks = dim3(50);
		}

		CubelistContext::CubelistContext(Cube* cubes, uint32_t num_cubes, const std::string& map_name)
		{
			m_map_name = map_name;
			m_d_cubes = cubes;
			m_number_of_cubes = num_cubes;
			m_default_prim = std::make_unique<Cuboid>(glm::vec4(0.f, 0.f, 0.f, 1.f),
				glm::vec3(0.f, 0.f, 0.f),
				glm::vec3(1.f, 1.f, 1.f));
			m_threads_per_block = dim3(cMAX_THREADS_PER_BLOCK);
			m_num_blocks = dim3(50);
		}

		Cube* CubelistContext::getCubesDevicePointer() const
		{
			return m_d_cubes;
		}

		void CubelistContext::setCubesDevicePointer(Cube* cubes)
		{
			m_d_cubes = cubes;
		}

		uint32_t CubelistContext::getNumberOfCubes() const
		{
			return m_number_of_cubes;
		}

		void CubelistContext::setNumberOfCubes(uint32_t numberOfCubes)
		{
			m_number_of_cubes = numberOfCubes;
		}

		void CubelistContext::unmapCubesShm()
		{
			if (m_d_cubes == nullptr)
				return;

			cudaIpcCloseMemHandle(m_d_cubes);
			m_d_cubes = nullptr;
		}

		void CubelistContext::updateVBOOffsets()
		{
			thrust::exclusive_scan(m_num_voxels_per_type.begin(), m_num_voxels_per_type.end(), m_vbo_offsets.begin());
			m_d_vbo_offsets = m_vbo_offsets;
		}

		void CubelistContext::updateCudaLaunchVariables(Vector3ui supervoxel_size)
		{
			m_threads_per_block = dim3(cMAX_THREADS_PER_BLOCK);
			m_num_blocks = dim3(m_number_of_cubes / cMAX_THREADS_PER_BLOCK + 1);
		}
	}
}