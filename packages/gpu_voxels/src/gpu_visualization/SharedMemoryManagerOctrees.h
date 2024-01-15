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
 * \date    2014-07-09
 *
 * \brief This class is for the management of the interprocess communication with the provider.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGEROCTREES_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGEROCTREES_H_INCLUDED

#include <memory>

#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_visualization/SharedMemoryManager.h>

namespace gpu_voxels
{
	struct Cube;
}

typedef int8_t Probability;

namespace gpu_voxels
{
	namespace visualization
	{
		
		class SharedMemoryManager;
		class SharedMemoryManagerOctrees
		{
		public:

			SharedMemoryManagerOctrees();

			[[nodiscard]] uint32_t getNumberOfOctreesToDraw() const;

			[[nodiscard]] std::string getNameOfOctree(uint32_t index) const;

			[[nodiscard]] bool getOctreeVisualizationData(Cube*& cubes, uint32_t& size, uint32_t index) const;

			void setView(const Vector3ui& start_voxel, const Vector3ui& end_voxel);

			void setOctreeBufferSwappedToFalse(uint32_t index);
			[[nodiscard]] bool hasOctreeBufferSwapped(uint32_t index) const;

			void setOctreeOccupancyThreshold(uint32_t index, Probability threshold);

			[[nodiscard]] bool getSuperVoxelSize(uint32_t& sdim) const;
			void setSuperVoxelSize(uint32_t sdim);

		private:

			std::unique_ptr<SharedMemoryManager> shmm;
		}
		;
	} //end of namespace visualization
} //end of namespace gpu_voxels
#endif /* GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGEROCTREES_H_INCLUDED */
