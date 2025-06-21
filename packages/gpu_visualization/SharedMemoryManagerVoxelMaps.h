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
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELMAPS_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELMAPS_H_INCLUDED

#include <memory>

#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>
//#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_visualization/SharedMemoryManager.h>

namespace gpu_voxels {
	namespace visualization {

		class SharedMemoryManager;

		class SharedMemoryManagerVoxelMaps
		{
		public:

			SharedMemoryManagerVoxelMaps();
			~SharedMemoryManagerVoxelMaps();

			uint32_t getNumberOfVoxelMapsToDraw() const;

			bool getDevicePointer(void*& handler, uint32_t index) const;
			bool getVoxelMapDimension(Vector3ui& dim, uint32_t index) const;
			bool getVoxelMapSideLength(float& voxel_side_length, uint32_t index) const;
			bool getVoxelMapName(std::string& map_name, uint32_t index) const;
			void setVoxelMapDataChangedToFalse(uint32_t index) const;
			bool hasVoxelMapDataChanged(uint32_t index) const;
			bool getVoxelMapType(MapType& type, uint32_t index) const;

		private:
			std::unique_ptr<SharedMemoryManager> shmm;
		};
	} //end of namespace visualization
} //end of namespace gpu_voxels
#endif /* GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELMAPS_H_INCLUDED */