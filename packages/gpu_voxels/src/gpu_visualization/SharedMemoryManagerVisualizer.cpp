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
#include <gpu_visualization/SharedMemoryManagerVisualizer.h>
#include <gpu_visualization/SharedMemoryManager.h>

#include <glm/vec3.hpp>

namespace gpu_voxels {
    namespace visualization {

        SharedMemoryManagerVisualizer::SharedMemoryManagerVisualizer()
	        : shmm(std::make_unique<SharedMemoryManager>(shm_segment_name_visualizer, true))
        {}

        bool SharedMemoryManagerVisualizer::getCameraTargetPoint(glm::vec3& target) const
        {
            const auto [targetPoint, amountFound] = shmm->getMemSegment().find<Vector3f>(shm_variable_name_target_point.c_str());
            if (amountFound == 0)
                return false;

            const Vector3f& t = *targetPoint;
            target = glm::vec3(t.x(), t.y(), t.z());
            return true;
        }

        DrawTypes SharedMemoryManagerVisualizer::getDrawTypes() const
        {
            const auto [drawType, amountFound] = shmm->getMemSegment().find<DrawTypes>(shm_variable_name_set_draw_types.c_str());
            if (amountFound == 0)
                return {};
            
            return *drawType;
        }
    } //end of namespace visualization
} //end of namespace gpu_voxels