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
#ifndef GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVISUALIZER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVISUALIZER_H_INCLUDED

#include <memory>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>

#include <glm/fwd.hpp>

namespace gpu_voxels
{
	namespace visualization
	{
		class SharedMemoryManager;
		class SharedMemoryManagerVisualizer
		{
		public:

			SharedMemoryManagerVisualizer();

			[[nodiscard]] bool getCameraTargetPoint(glm::vec3& target) const;
			[[nodiscard]] DrawTypes getDrawTypes() const;

		private:

			std::unique_ptr<SharedMemoryManager> shmm;
		};
	} //end of namespace visualization
} //end of namespace gpu_voxels
#endif /* GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVISUALIZER_H_INCLUDED */