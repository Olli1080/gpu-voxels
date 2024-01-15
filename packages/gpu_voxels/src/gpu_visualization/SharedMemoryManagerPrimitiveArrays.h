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
 * \date    2015-01-06
 *
 * \brief This class is for the management of the interprocess communication with the provider.
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERPRIMITIVEARRAYS_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERPRIMITIVEARRAYS_H_INCLUDED

//#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>
#include <gpu_voxels/primitive_array/PrimitiveArray.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_visualization/logging/logging_visualization.h>
#include <gpu_visualization/SharedMemoryManager.h>

#include <memory>

namespace gpu_voxels {
	namespace visualization {

		class SharedMemoryManager;

		class SharedMemoryManagerPrimitiveArrays
		{
		public:

			SharedMemoryManagerPrimitiveArrays();
			~SharedMemoryManagerPrimitiveArrays();

			uint32_t getNumberOfPrimitiveArraysToDraw() const;
			bool getPrimitivePositions(const uint32_t index, Vector4f** d_positions, uint32_t& size, primitive_array::PrimitiveType& type) const;
			bool hasPrimitiveBufferChanged(const uint32_t index) const;
			void setPrimitiveBufferChangedToFalse(const uint32_t index);
			std::string getNameOfPrimitiveArray(const uint32_t index) const;

		private:

			std::unique_ptr<SharedMemoryManager> shmm;
		};
	} //end of namespace visualization
} //end of namespace gpu_voxels
#endif /* GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERPRIMITIVEARRAYS_H_INCLUDED */