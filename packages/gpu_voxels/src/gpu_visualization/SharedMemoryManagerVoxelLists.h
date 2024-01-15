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
* \date    2015-05-07
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELLISTS_H
#define GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELLISTS_H

#include <memory>
#include <string>

namespace gpu_voxels
{
	struct Cube;
}

namespace gpu_voxels
{
	namespace visualization
	{
		class SharedMemoryManager;
		class SharedMemoryManagerVoxelLists
		{
		public:

			SharedMemoryManagerVoxelLists();

			[[nodiscard]] uint32_t getNumberOfVoxelListsToDraw() const;

			[[nodiscard]] bool getVoxelListName(std::string& map_name, uint32_t index) const;

			[[nodiscard]] bool getVisualizationData(Cube*& cubes, uint32_t& size, uint32_t index) const;
			void setBufferSwappedToFalse(uint32_t index) const;
			[[nodiscard]] bool hasBufferSwapped(uint32_t index) const;

		private:
			std::unique_ptr<SharedMemoryManager> shmm;
		};
	} //end of namespace visualization
} //end of namespace gpu_voxels
#endif // GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELLISTS_H