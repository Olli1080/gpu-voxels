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
 * \date    2014-12-17
 *
 * \brief This file contains the CUDA-Kernels of the visualizer.
 *
 */
 //----------------------------------------------------------------------/*

#ifndef GPU_VOXELS_VISUALIZATION_KERNELS_VOXEL_MAP_VISUALIZER_OPERATIONS_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_KERNELS_VOXEL_MAP_VISUALIZER_OPERATIONS_H_INCLUDED

#include <cuda_runtime.h>
#include <assert.h>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>
//#include <gpu_voxels/voxelmap/Voxel.h>
//#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/voxellist/VoxelList.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_visualization/visualizerDefines.h>

namespace gpu_voxels {
	namespace visualization {

		//////////////////////////////////// CUDA device functions /////////////////////////////////////////
		
		//////////////////////////////////// CUDA kernel functions /////////////////////////////////////////

		__global__ void fill_vbo_without_precounting(ProbabilisticVoxel* voxelMap, Vector3ui dim_voxel_map,
			Vector3ui dim_super_voxel, Vector3ui start_voxel,
			Vector3ui end_voxel, Probability occupancy_threshold, float4* vbo,
			uint32_t* vbo_offsets, uint32_t* vbo_limits,
			uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes);

		__global__ void fill_vbo_without_precounting(const BitVectorVoxel* voxelMap, Vector3ui dim_voxel_map,
			Vector3ui dim_super_voxel, Vector3ui start_voxel,
			Vector3ui end_voxel, uint8_t occupancy_threshold, float4* vbo,
			const uint32_t* vbo_offsets, const uint32_t* vbo_limits,
			uint32_t* write_index, const uint8_t* draw_voxel_type, const uint8_t* prefixes);

		__global__ void fill_vbo_without_precounting(DistanceVoxel* voxelMap, Vector3ui dim_voxel_map,
			Vector3ui dim_super_voxel, Vector3ui start_voxel,
			Vector3ui end_voxel, visualizer_distance_drawmodes drawmode, float4* vbo,
			uint32_t* vbo_offsets, uint32_t* vbo_limits,
			uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes);

		
		__global__ void fill_vbo_with_cubelist(Cube* cubes, uint32_t size, float4* vbo, uint32_t* vbo_offsets,
			uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes);
		
		__global__ void calculate_cubes_per_type_list(const Cube* cubes, uint32_t size, uint32_t* cubes_per_type,
			const uint8_t* draw_voxel_type, const uint8_t* prefixes);
		
		__global__ void find_cubes_by_coordinates(const Cube* cubes, size_t num_cubes, Vector3ui coords, Cube* found_cube, bool* found_flag);

		/*
		 *
		 *
		 *
		 */

		void fill_vbo_without_precounting_host(dim3 num_blocks, dim3 thread_per_block, ProbabilisticVoxel* voxelMap, Vector3ui dim_voxel_map,
			Vector3ui dim_super_voxel, Vector3ui start_voxel,
			Vector3ui end_voxel, Probability occupancy_threshold, float4* vbo,
			uint32_t* vbo_offsets, uint32_t* vbo_limits,
			uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes);

		void fill_vbo_without_precounting_host(dim3 num_blocks, dim3 thread_per_block, BitVectorVoxel* voxelMap, Vector3ui dim_voxel_map,
			Vector3ui dim_super_voxel, Vector3ui start_voxel,
			Vector3ui end_voxel, uint8_t occupancy_threshold, float4* vbo,
			uint32_t* vbo_offsets, uint32_t* vbo_limits,
			uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes);

		void fill_vbo_without_precounting_host(dim3 num_blocks, dim3 thread_per_block, DistanceVoxel* voxelMap, Vector3ui dim_voxel_map,
			Vector3ui dim_super_voxel, Vector3ui start_voxel,
			Vector3ui end_voxel, visualizer_distance_drawmodes drawmode, float4* vbo,
			uint32_t* vbo_offsets, uint32_t* vbo_limits,
			uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes);
		
		void fill_vbo_with_cubelist_host(dim3 num_blocks, dim3 thread_per_block, Cube* cubes, uint32_t size, float4* vbo, uint32_t* vbo_offsets,
			uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes);
		
		void calculate_cubes_per_type_list_host(dim3 num_blocks, dim3 thread_per_block, Cube* cubes, uint32_t size, uint32_t* cubes_per_type,
			uint8_t* draw_voxel_type, uint8_t* prefixes);
		
		void find_cubes_by_coordinates_host(dim3 num_blocks, dim3 thread_per_block, const Cube* cubes, size_t num_cubes, Vector3ui coords, Cube* found_cube, bool* found_flag);
	} // end of ns
} // end of ns

#endif
