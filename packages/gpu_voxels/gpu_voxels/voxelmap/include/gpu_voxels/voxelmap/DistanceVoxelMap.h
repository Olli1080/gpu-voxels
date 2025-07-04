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
 * \author  Christian Juelg
 * \date    2015-08-19
 *
 */
//----------------------------------------------------------------------
#ifndef DISTANCEVOXELMAP_H
#define DISTANCEVOXELMAP_H

#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include <gpu_voxels/voxel/DistanceVoxel.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>

#include <gpu_voxels/voxel/DistanceVoxel.hpp>

typedef gpu_voxels::DistanceVoxel::extract_byte_distance::free_space_t free_space_t;
typedef gpu_voxels::DistanceVoxel::init_floodfill_distance::manhattan_dist_t manhattan_dist_t;

namespace gpu_voxels {
    namespace voxelmap {

        typedef unsigned int uint;

        class DistanceVoxelMap : public TemplateVoxelMap<DistanceVoxel>
        {
        public:

            typedef DistanceVoxel Voxel;
            typedef TemplateVoxelMap<Voxel> Base;

            DistanceVoxelMap(Vector3ui dim, float voxel_side_length, MapType map_type);
            DistanceVoxelMap(Voxel* dev_data, Vector3ui dim, float voxel_side_length, MapType map_type);
            ~DistanceVoxelMap() override = default;

            MapType getTemplateType() const override { return MT_DISTANCE_VOXELMAP; }

            virtual size_t collideWithTypes(GpuVoxelsMapSharedPtr other, BitVectorVoxel& meanings_in_collision, float coll_threshold = 1.f, const Vector3ui& offset = Vector3ui::Zero());

            

            bool insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud* robot_links,
                const std::vector<BitVoxelMeaning>& voxel_meanings = {},
                const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks = {},
                BitVector<BIT_VECTOR_LENGTH>* colliding_meanings = nullptr) override;

            void clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning) override;

            //protected:
            //  virtual void clearVoxelMapRemoteLock(const uint32_t bit_index);

        public:

            virtual bool mergeOccupied(std::shared_ptr<ProbVoxelMap> other, const Vector3ui& voxel_offset = Vector3ui::Zero(), float occupancy_threshold = 0.5f);

            void jumpFlood3D(int block_size = cMAX_THREADS_PER_BLOCK, int debug = 0, bool logging_reinit = false);
            void exactDistances3D(std::vector<Vector3f>& points);
            void parallelBanding3D(uint32_t m1 = 1, uint32_t m2 = 1, uint32_t m3 = 1, 
                uint32_t m1_blocksize = gpu_voxels::PBA_DEFAULT_M1_BLOCK_SIZE, 
                uint32_t m2_blocksize = gpu_voxels::PBA_DEFAULT_M2_BLOCK_SIZE, 
                uint32_t m3_blocksize = gpu_voxels::PBA_DEFAULT_M3_BLOCK_SIZE, 
                bool detailtimer = false);

            void fill_pba_uninit(DistanceVoxelMap& other);
            void fill_pba_uninit();

            DistanceVoxel::pba_dist_t getSquaredObstacleDistance(const Vector3ui& pos) const;
            DistanceVoxel::pba_dist_t getSquaredObstacleDistance(uint x, uint y, uint z) const;
            DistanceVoxel::pba_dist_t getObstacleDistance(const Vector3ui& pos) const;
            DistanceVoxel::pba_dist_t getObstacleDistance(uint x, uint y, uint z) const;

            void getSquaredDistancesToHost(const std::vector<uint>& indices, std::vector<DistanceVoxel::pba_dist_t>& output);
            void getSquaredDistances(thrust::device_ptr<uint> dev_indices_begin, thrust::device_ptr<uint> dev_indices_end, thrust::device_ptr<DistanceVoxel::pba_dist_t> dev_output);

            void getDistancesToHost(std::vector<uint>& indices, std::vector<DistanceVoxel::pba_dist_t>& output);
            void getDistances(thrust::device_ptr<uint> dev_indices_begin, thrust::device_ptr<uint> dev_indices_end, thrust::device_ptr<DistanceVoxel::pba_dist_t> dev_output);

            void extract_distances(free_space_t* dev_distances, int robot_radius) const;
            void init_floodfill(free_space_t* dev_distances, manhattan_dist_t* dev_manhattan_distances, uint robot_radius);

            DistanceVoxel::accumulated_diff differences3D(std::shared_ptr<DistanceVoxelMap> other_map, int debug = 0, bool logging_reinit = true);
        };

        struct mergeOccupiedOperator
        {
            typedef thrust::tuple<ProbabilisticVoxel, uint> inputTuple;

            Vector3ui map_dim;
            Vector3ui offset;

            mergeOccupiedOperator(const Vector3ui& ref_map_dim, const Vector3ui& coord_offset)
            {
                offset = coord_offset;
                map_dim = ref_map_dim;
            }

            __host__ __device__
            DistanceVoxel operator()(const inputTuple& input) const
            {
	            const uint index = thrust::get<1>(input);

                // get int coords of voxel; use map_dim
                const Vector3ui coords = mapToVoxels(index, map_dim);

                // add offset
                return { coords + offset };
            }
        };

        struct probVoxelOccupied
        {
            typedef thrust::tuple<ProbabilisticVoxel, uint> inputTuple;
            Probability occ_threshold;

            probVoxelOccupied(Probability occ_threshold_)
            {
                occ_threshold = occ_threshold_;
            }

            __host__ __device__
            bool operator()(const inputTuple& input) const
            {
                return thrust::get<0>(input).getOccupancy() > occ_threshold;
            }
        };

    } // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif // DISTANCEVOXELMAP_H