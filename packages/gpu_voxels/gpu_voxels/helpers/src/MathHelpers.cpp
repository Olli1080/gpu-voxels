// this is for emacs file handling -&- mode: c++; indent-tabs-mode: nil -&-

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
 * \date    2015-10-02
 *
 */
//----------------------------------------------------------------------

#include "MathHelpers.h"

#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>

#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels
{
    void computeLinearLoad(uint32_t nr_of_items, uint32_t& blocks, uint32_t& threads_per_block)
    {
        //  if (nr_of_items <= cMAX_NR_OF_BLOCKS)
        //  {
        //    *blocks = nr_of_items;
        //    *threads_per_block = 1;
        //  }
        //  else
        //  {

        if (nr_of_items == 0)
        {
            LOGGING_WARNING(
                Gpu_voxels_helpers,
                "Number of Items is 0. Blocks and Threads per Block is set to 1. Size 0 would lead to a Cuda ERROR" << endl);

            blocks = 1;
            threads_per_block = 1;
            return;
        }

        if (nr_of_items <= cMAX_NR_OF_BLOCKS * cMAX_THREADS_PER_BLOCK)
        {
            blocks = (nr_of_items + cMAX_THREADS_PER_BLOCK - 1)
                / cMAX_THREADS_PER_BLOCK;                          // calculation replaces a ceil() function
            threads_per_block = cMAX_THREADS_PER_BLOCK;
        }
        else
        {
            /* In this case the kernel must perform multiple runs because
             * nr_of_items is larger than the gpu can handle at once.
             * To overcome this limit, use standard parallelism offsets
             * as when programming host code (increment by the number of all threads
             * running). Use something like
             *
             *   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
             *
             *   while (i < nr_of_items)
             *   {
             *     // perform some kernel operations here
             *
             *     // increment by number of all threads that are running
             *     i += blockDim.x * gridDim.x;
             *   }
             *
             * CAUTION: currently cMAX_NR_OF_BLOCKS is 64K, although
             *          GPUs with SM >= 3.0 support up to 2^31 -1 blocks in a grid!
             */
            LOGGING_ERROR(
                Gpu_voxels_helpers,
                "computeLinearLoad: Number of Items " << nr_of_items << " exceeds the limit cMAX_NR_OF_BLOCKS * cMAX_THREADS_PER_BLOCK = " << (cMAX_NR_OF_BLOCKS * cMAX_THREADS_PER_BLOCK) << "! This number of items cannot be processed in a single invocation." << endl);
            blocks = cMAX_NR_OF_BLOCKS;
            threads_per_block = cMAX_THREADS_PER_BLOCK;
        }
    }
} // end of namespace