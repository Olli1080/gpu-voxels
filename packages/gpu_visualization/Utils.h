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
 * \brief This file contains some helper functions.
 *
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_UTILS_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_UTILS_H_INCLUDED


#include <gpu_visualization/Camera.h>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {
	namespace visualization {

		void ExitOnGLError(const std::string&);

		std::string typeToString(BitVoxelMeaning type);
		std::string typeToString(MapType type);

		// data format converter functions
		__inline__ glm::vec3 convertFromVector3uiToVec3(gpu_voxels::Vector3ui v)
		{
			glm::vec3 r;
			r.x = static_cast<float>(v.x());
			r.y = static_cast<float>(v.y());
			r.z = static_cast<float>(v.z());
			return r;
		}

		__inline__ gpu_voxels::Vector3ui convertFromVec3ToVector3ui(const glm::vec3& v)
		{
			gpu_voxels::Vector3ui r;
			r.x() = static_cast<uint32_t>(v.x);
			r.y() = static_cast<uint32_t>(v.y);
			r.z() = static_cast<uint32_t>(v.z);
			return r;
		}

	} // end of namespace visualization
} // end of namespace gpu_voxels

#endif