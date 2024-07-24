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

#include "SharedMemoryManagerVoxelLists.h"

#include <gpu_visualization/SharedMemoryManager.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>

namespace gpu_voxels {
    namespace visualization {

        SharedMemoryManagerVoxelLists::SharedMemoryManagerVoxelLists()
            : shmm(std::make_unique<SharedMemoryManager>(shm_segment_name_voxellists, true))
        {}

        uint32_t gpu_voxels::visualization::SharedMemoryManagerVoxelLists::getNumberOfVoxelListsToDraw() const
        {
            const std::pair<uint32_t*, std::size_t> res = shmm->getMemSegment().find<uint32_t>(shm_variable_name_number_of_voxellists.c_str());
            if (res.second == 0)
                return 0;
            
            return *res.first;
        }

        bool SharedMemoryManagerVoxelLists::getVoxelListName(std::string& list_name, uint32_t index) const
        {
            const std::string index_str = std::to_string(index);
            const std::string voxel_list_name_var_name = shm_variable_name_voxellist_name + index_str;

            const auto [rawName, strLength] = shmm->getMemSegment().find<char>(voxel_list_name_var_name.c_str());
            if (strLength == 0)
            { /*If the segment couldn't be find or the string is empty*/
                list_name = "voxellist_" + index_str;
                return false;
            }
            list_name.assign(rawName, strLength);
            return true;
        }

        bool SharedMemoryManagerVoxelLists::getVisualizationData(Cube*& cubes, uint32_t& size, uint32_t index) const
        {
            const std::string handler_name = shm_variable_name_voxellist_handler_dev_pointer + std::to_string(index);
            const std::string number_cubes_name = shm_variable_name_voxellist_num_voxels + std::to_string(index);

            // Find shared memory handles for: Cubes device pointer, number of cubes
            const auto [cubeHandle, c_res_size] = shmm->getMemSegment().find<cudaIpcMemHandle_t>(handler_name.c_str());
            const auto [numberOfCubes, noc_res_size] = shmm->getMemSegment().find<uint32_t>(number_cubes_name.c_str());

            if (c_res_size == 0 || noc_res_size == 0)
            {
                // Shared memory handles not found
                return false;
            }

            const uint32_t new_size = *numberOfCubes;
            if (new_size > 0)
            {
                Cube* new_cubes;
                const cudaError_t cuda_error = cudaIpcOpenMemHandle(reinterpret_cast<void**>(&new_cubes), *cubeHandle, cudaIpcMemLazyEnablePeerAccess);
                if (cuda_error == cudaSuccess)
                {
                    cubes = new_cubes;
                    size = new_size;
                }
                else
                {
                    // IPC handle to device pointer could not be opened
                    cudaIpcCloseMemHandle(new_cubes);
                    return false;
                }
            }
            else
            {
                cubes = nullptr; // No memory is allocated when voxellist is empty
                size = new_size;
            }

            return true;
        }

        void SharedMemoryManagerVoxelLists::setBufferSwappedToFalse(uint32_t index) const
        {
            const std::string swapped_buffer_name = shm_variable_name_voxellist_buffer_swapped + std::to_string(index);
            const auto [isBufferSwapped, res_size] = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());

            if (res_size != 0)
                *isBufferSwapped = false;
        }

        bool SharedMemoryManagerVoxelLists::hasBufferSwapped(uint32_t index) const
        {
            const std::string index_str = std::to_string(index);
            const std::string swapped_buffer_name = shm_variable_name_voxellist_buffer_swapped + index_str;
            const auto [isBufferSwapped, res_size] = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());

            if (res_size != 0)
                return *isBufferSwapped;

            return false;
        }
    } //end of namespace visualization
} //end of namespace gpu_voxels