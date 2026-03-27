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
 * \date    2017-04-09
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/helpers_highlevel/DistanceMapConverter.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxelmap/DistanceVoxelMap.h>

#if defined(__INTELLISENSE___) || defined(__RESHARPER__) 
// in here put whatever is your favorite flavor of intellisense workarounds
#ifndef __CUDACC__ 
#define __CUDACC__
#include <device_functions.h>
#include "device_launch_parameters.h"
#endif
#endif

namespace gpu_voxels
{
    namespace distance_map_converter
    {
        using namespace voxellist;

        GVL_GLOBAL
            void kernelConvertToBitVectorVoxellist(const free_space_t* free_space_list, const MapVoxelID* voxel_id_list, size_t num_elemets, const Vector3ui dims,
                MapVoxelID* ret_voxel_id, Vector3ui* ret_coords, BitVectorVoxel* ret_voxel)
        {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elemets;
                i += blockDim.x * gridDim.x)
            {

                // cast to larger range, to prevent overflow
                uint16_t free_space = free_space_list[i] + eBVM_SWEPT_VOLUME_START; // Offset


                BitVectorVoxel new_voxel;
                new_voxel.bitVector().setBit(eBVM_OCCUPIED);

                // upper cap
                if (free_space > (eBVM_SWEPT_VOLUME_END + 1))
                {
                    new_voxel.bitVector().setBit(eBVM_UNDEFINED);
                    free_space = (eBVM_SWEPT_VOLUME_END + 1);
                }

                for (size_t sv_id = eBVM_SWEPT_VOLUME_START; sv_id < free_space; ++sv_id)
                {
                    new_voxel.bitVector().setBit(sv_id);
                }


                Vector3ui pos;
                MapVoxelID linear_id = voxel_id_list[i];
                MapVoxelID linear_id_tmp = linear_id;
                pos.z() = linear_id_tmp / (dims.x() * dims.y());
                pos.y() = (linear_id_tmp -= pos.z() * (dims.x() * dims.y())) / dims.x();
                pos.x() = (linear_id_tmp -= pos.y() * dims.x());


                ret_coords[i] = pos;
                ret_voxel[i] = new_voxel;
                ret_voxel_id[i] = linear_id;
            }
        }
        
        /*!
         * \brief The transform_to_bitvoxel struct
         * This takes a distance voxel and generates a bitvoxel from it.
         * Therefore it calculates the distance voxels clearance and
         * maps it to the SweptVolume Bits:
         * No free space: No SV-ID set.
         * 1 Unit free space: SV-ID 1 set.
         * 2 Units free space: SV-ID 1 + 2 set.
         * ...
         * 255 Units free space: All SV-IDs set.
         * More than 255 Units free: All SV-IDs set + Undefined Bit set.
         *
         */
        struct transform_to_bitvoxel : parallel::tuple<MapVoxelID, Vector3ui, BitVectorVoxel>
        {
            typedef parallel::tuple<MapVoxelID, Vector3ui, BitVectorVoxel> keyCoordVoxelTriple;
            typedef parallel::tuple<free_space_t, MapVoxelID > dist_tuple_t;

            Vector3ui dims;


            GVL_HOST_DEVICE
                transform_to_bitvoxel(Vector3ui dims_) :
                dims(dims_) {}

            GVL_HOST_DEVICE
                keyCoordVoxelTriple operator()(const dist_tuple_t& tuple) const {
                keyCoordVoxelTriple ret_triple;

                // get pos from zipiterator/tuple
                uint16_t free_space = parallel::get<0>(tuple); // cast to larger range, to prevent overflow
                MapVoxelID linear_id = parallel::get<1>(tuple);

                // pos is the position of the voxel dv
                Vector3ui pos;
                MapVoxelID linear_id_tmp = linear_id;
                pos.z() = linear_id_tmp / (dims.x() * dims.y());
                pos.y() = (linear_id_tmp -= pos.z() * (dims.x() * dims.y())) / dims.x();
                pos.x() = (linear_id_tmp -= pos.y() * dims.x());

                free_space += eBVM_SWEPT_VOLUME_START; // Offset


                BitVectorVoxel ret_voxel;
                ret_voxel.bitVector().setBit(eBVM_OCCUPIED);

                // upper cap
                if (free_space > (eBVM_SWEPT_VOLUME_END + 1))
                {
                    ret_voxel.bitVector().setBit(eBVM_UNDEFINED);
                    free_space = (eBVM_SWEPT_VOLUME_END + 1);
                }

                for (uint32_t sv_id = eBVM_SWEPT_VOLUME_START; sv_id < free_space; ++sv_id)
                {
                    ret_voxel.bitVector().setBit(sv_id);
                }

                parallel::get<0>(ret_triple) = linear_id;
                parallel::get<1>(ret_triple) = pos;
                parallel::get<2>(ret_triple) = ret_voxel;

                return ret_triple;
            }
        };
        
        size_t extract_given_distances(const voxelmap::DistanceVoxelMap& dist_map,
            free_space_t min_dist, free_space_t max_dist,
            BitVectorVoxelList& result) {

            // Step 1: Create an Vector containing the free-space distances of all voxels:
            parallel::device_vector<free_space_t> distances(dist_map.getVoxelMapSize());
            dist_map.extract_distances(parallel::raw_pointer_cast(distances.data()), 0);
            // Step 2: Count distances that match criteria and allocate a Bitvecor-Voxellist of that length
            size_t num_matching_voxels = parallel::count_if(distances.begin(), distances.end(), in_range(min_dist, max_dist));
            result.resize(num_matching_voxels);
            parallel::device_vector<free_space_t> matching_distances_dists(num_matching_voxels);
            parallel::device_vector<MapVoxelID> matching_distances_ids(num_matching_voxels);
            // Step 3: Copy matching Distance voxels into two new vectors
            parallel::counting_iterator<MapVoxelID> count_start(0);
            parallel::copy_if(parallel::make_zip_iterator(parallel::make_tuple(distances.begin(), count_start)),
                parallel::make_zip_iterator(parallel::make_tuple(distances.end(), count_start + dist_map.getVoxelMapSize())),
                parallel::make_zip_iterator(parallel::make_tuple(matching_distances_dists.begin(), matching_distances_ids.begin())),
                in_range_tuple(min_dist, max_dist));
            // Step 4: Transform distances and IDs into a Voxellist
            parallel::transform(parallel::make_zip_iterator(parallel::make_tuple(matching_distances_dists.begin(), matching_distances_ids.begin())),
                parallel::make_zip_iterator(parallel::make_tuple(matching_distances_dists.end(), matching_distances_ids.end())),
                result.getBeginTripleZipIterator(),
                transform_to_bitvoxel(dist_map.getDimensions()));

            //  uint32_t num_blocks, threads_per_block;
            //  computeLinearLoad(num_matching_voxels, &num_blocks, &threads_per_block);
            //  GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
            //  kernelConvertToBitVectorVoxellist<<<num_blocks, threads_per_block>>>(parallel::raw_pointer_cast(matching_distances_dists.data()),
            //                                                                       parallel::raw_pointer_cast(matching_distances_ids.data()),
            //                                                                       num_matching_voxels, dist_map.getDimensions(),
            //                                                                       result.getDeviceIdPtr(), result.getDeviceCoordPtr(), result.getDeviceDataPtr());
            //  GVL_CHECK_ERROR();
            GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());

            return num_matching_voxels;
        }

    }
}