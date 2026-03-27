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
 * \author  Florian Drews
 * \date    2013-11-07
 *
 */
 //----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_VOXEL_H_INCLUDED
#define GPU_VOXELS_OCTREE_VOXEL_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.hpp>
#include <gpu_voxels/helpers/common_defines.h>

#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/Nodes.h>

#include <assert.h>

namespace gpu_voxels {
	namespace NTree {

		class Voxel
		{
		public:
			// TODO choose better memory layout; may use one byte of voxel_id for occ. probab.

			OctreeVoxelID voxelId;
			Vector3ui coordinates;

			GVL_HOST_DEVICE
				friend bool operator<(Voxel a, Voxel b)
			{
				return a.voxelId < b.voxelId; // | (a.voxel_id == b.voxel_id & a.occupation < b.occupation);
			}

			GVL_HOST_DEVICE
				friend bool operator==(Voxel a, Voxel b)
			{
				return a.voxelId == b.voxelId && a.coordinates == b.coordinates && a.occupancy == b.occupancy;
			}

		private:
			Probability occupancy;

		public:

			GVL_HOST_DEVICE
				Voxel()
			{
			}

			GVL_HOST_DEVICE
				Voxel(OctreeVoxelID voxelID, Vector3ui coordinates, Probability occupancy)
			{
				this->voxelId = voxelID;
				this->coordinates = coordinates;
				this->occupancy = occupancy;
			}

			GVL_HOST_DEVICE
				__forceinline__
				Probability getOccupancy() const
			{
				return occupancy;
			}

			GVL_HOST_DEVICE
				__forceinline__
				void setOccupancy(Probability value)
			{
				occupancy = value;
			}

		};

		struct count_per_size
		{
			OctreeVoxelID m_cube_side_length;

			GVL_HOST_DEVICE
				count_per_size(OctreeVoxelID cube_side_length)
			{
				m_cube_side_length = cube_side_length;
			}

			GVL_HOST_DEVICE
			voxel_count operator()(Cube value)
			{
				return value.m_side_length == m_cube_side_length;
			}
		};

	}
}

#endif /* VOXEL_H_ */
