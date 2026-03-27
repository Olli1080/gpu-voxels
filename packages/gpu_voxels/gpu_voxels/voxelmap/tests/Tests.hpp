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
 * \date    2016-05-25
 *
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_TESTS_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_TESTS_HPP_INCLUDED


#include <gpu_voxels/voxelmap/Tests.h>
#include <gpu_voxels/voxelmap/TemplateVoxelMap.hpp>
#include <gpu_voxels/voxelmap/kernels/VoxelMapTests.hpp>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

namespace gpu_voxels {
	namespace voxelmap {
		namespace test {

			struct RandGen
			{
				parallel::uniform_real_distribution<float> dist_x;
				parallel::uniform_real_distribution<float> dist_y;
				parallel::uniform_real_distribution<float> dist_z;

				GVL_HOST_DEVICE
					RandGen(Vector3f min, Vector3f max) {
					// create a uniform_real_distribution to produce floats
					dist_x = parallel::uniform_real_distribution<float>(min.x, max.x);
					dist_y = parallel::uniform_real_distribution<float>(min.y, max.y);
					dist_z = parallel::uniform_real_distribution<float>(min.z, max.z);
				}

				GVL_HOST_DEVICE
					Vector3f operator() (const unsigned int n)
				{
					// create a minstd_rand object to act as our source of randomness
					parallel::minstd_rand rnd;
					rnd.discard(n);
					return { dist_x(rnd), dist_y(rnd), dist_z(rnd) };
				}
			};

			template<class Voxel>
			void triggerAddressingTest(Vector3ui dimensions, float voxel_side_length,
				size_t nr_of_tests, bool* success)
			{
				parallel::device_vector<Vector3f> dev_testpoint_list(nr_of_tests);

				srand(time(nullptr));
				Voxel* voxelmap_base_adress = (Voxel*)1234;

				bool* dev_success;
				GVL_HANDLE_ERROR(GVL_MALLOC(&dev_success, sizeof(bool)));
				GVL_HANDLE_ERROR(GVL_MEMCPY(dev_success, success, sizeof(bool), GVL_MEMCPY_HOST_TO_DEVICE));

				const RandGen myRandGen(Vector3f::Zero(), dimensions * voxel_side_length);

				const parallel::counting_iterator<unsigned int> index_sequence_begin(0);
				parallel::transform(index_sequence_begin, index_sequence_begin + nr_of_tests, dev_testpoint_list.begin(), myRandGen);

				uint32_t num_blocks;
				uint32_t threads_per_block;
				computeLinearLoad(nr_of_tests, &num_blocks, &threads_per_block);
				kernelAddressingTest << < num_blocks, threads_per_block >> > (voxelmap_base_adress, dimensions, voxel_side_length,
					parallel::raw_pointer_cast(dev_testpoint_list.data()),
					nr_of_tests, dev_success);
				GVL_CHECK_ERROR();
				GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
				GVL_HANDLE_ERROR(GVL_MEMCPY(success, dev_success, sizeof(bool), GVL_MEMCPY_DEVICE_TO_HOST));
			}

		} // end of namespace
	} // end of namespace
} // end of namespace

#endif