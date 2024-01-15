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
 * \brief   Saves all necessary stuff to draw the cube list of an
 * octree or a voxellist.
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_CUBELISTCONTEXT_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_CUBELISTCONTEXT_H_INCLUDED

#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_visualization/DataContext.h>

namespace gpu_voxels {
	namespace visualization {

		typedef std::pair<glm::vec4, glm::vec4> colorPair;

		class CubelistContext : public DataContext
		{
		public:

			CubelistContext(const std::string& map_name);
			CubelistContext(Cube* cubes, uint32_t num_cubes, const std::string& map_name);

			~CubelistContext() override = default;

			[[nodiscard]] Cube* getCubesDevicePointer() const;
			void setCubesDevicePointer(Cube* cubes);

			[[nodiscard]] uint32_t getNumberOfCubes() const;
			void setNumberOfCubes(uint32_t numberOfCubes);

			void unmapCubesShm();

			void updateVBOOffsets() override;

			void updateCudaLaunchVariables(Vector3ui supervoxel_size = Vector3ui::Ones()) override;

		private:
			// the GPU pointer to the cubes of this context
			Cube* m_d_cubes;
			// the number of cubes in m_cubes
			uint32_t m_number_of_cubes;
		};
	} // end of namespace visualization
} // end of namespace gpu_voxels
#endif