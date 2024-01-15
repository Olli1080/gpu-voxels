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
* \date    2015-05-05
*
*/
//----------------------------------------------------------------------

#include "VoxelListOperations.hpp"

namespace gpu_voxels
{
    namespace voxellist
    {
        __global__
        void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, BitVectorVoxel* this_voxel_list, uint32_t this_list_size,
            const ProbabilisticVoxel* other_map, Vector3ui other_map_dim, float col_threshold,
            Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* bitvoxel_results)
        {
            // NOP
            printf("kernelCollideWithVoxelMap not implemented for Octreee!");
        }

        __global__
        void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, BitVectorVoxel* this_voxel_list, uint32_t this_list_size,
            const BitVectorVoxel* other_map, Vector3ui other_map_dim,
            Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* bitvoxel_results)
        {
            // NOP
            printf("kernelCollideWithVoxelMap not implemented for Octreee!");
        }

        __global__
        void kernelCollideWithVoxelMapBitMask(const OctreeVoxelID* this_id_list, BitVectorVoxel* this_voxel_list, uint32_t this_list_size,
            const ProbabilisticVoxel* other_map, Vector3ui other_map_dim, float col_threshold,
            Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results)
        {
            // NOP
            printf("kernelCollideWithVoxelMapBitMask not implemented for Octreee!");
        }

        __global__
        void kernelCollideWithVoxelMapBitMask(const OctreeVoxelID* this_id_list, BitVectorVoxel* this_voxel_list, uint32_t this_list_size,
            const BitVectorVoxel* other_map, Vector3ui other_map_dim,
            Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results)
        {
            // NOP
            printf("kernelCollideWithVoxelMapBitMask not implemented for Octreee!");
        }

    }
}