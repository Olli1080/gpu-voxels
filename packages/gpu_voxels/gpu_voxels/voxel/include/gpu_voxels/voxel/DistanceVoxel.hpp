// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2015 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Christian Juelg
 * \date    2015-08-18
 *
 */
 //----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXELMAP_DISTANCE_VOXEL_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_DISTANCE_VOXEL_HPP_INCLUDED

#include "DistanceVoxel.h"

#include <Eigen/Dense>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {

	__host__ __device__
	inline Vector3ui DistanceVoxel::getObstacle() const
	{
		const uint x = m_obstacle & 0x3ff; //1023
		const uint y = (m_obstacle >> 10) & 0x3ff;
		const uint z = m_obstacle >> 20;

		return Vector3ui(x, y, z);
	}

	__host__ __device__
	inline DistanceVoxel::pba_dist_t DistanceVoxel::squaredObstacleDistance(Vector3i this_position) const
	{
		const Vector3ui obstacle = getObstacle();

		if (this_position.x() == PBA_UNINITIALISED_COORD
			|| this_position.y() == PBA_UNINITIALISED_COORD
			|| this_position.z() == PBA_UNINITIALISED_COORD
			|| obstacle.x() == PBA_UNINITIALISED_COORD
			|| obstacle.y() == PBA_UNINITIALISED_COORD
			|| obstacle.z() == PBA_UNINITIALISED_COORD
			)
		{
			return MAX_OBSTACLE_DISTANCE;
		}

		return static_cast<pba_dist_t>((this_position.cast<int64_t>() - obstacle.cast<int64_t>()).array().square().sum());
	}

	__host__ __device__
	inline DistanceVoxel::DistanceVoxel() = default; //default-initialise

	__host__ __device__
	inline DistanceVoxel::DistanceVoxel(const Vector3ui& o) {
		m_obstacle = o.x();
		m_obstacle |= o.y() << 10;
		m_obstacle |= o.z() << 20;
	}

	__host__ __device__
	inline DistanceVoxel::DistanceVoxel(const pba_voxel_t o) {
		m_obstacle = o;
	}

	__host__ __device__
	inline DistanceVoxel::DistanceVoxel(const uint x, const uint y, const uint z) {
		m_obstacle = x;
		m_obstacle |= y << 10;
		m_obstacle |= z << 20;
	}

	__host__ __device__
	inline DistanceVoxel::DistanceVoxel(const uint3& o) {
		m_obstacle = o.x;
		m_obstacle |= o.y << 10;
		m_obstacle |= o.z << 20;
	}

	__host__ __device__
	inline void DistanceVoxel::setObstacle(const Vector3ui& o) {
		m_obstacle = o.x();
		m_obstacle |= o.y() << 10;
		m_obstacle |= o.z() << 20;
	}

	__host__ __device__
	inline void DistanceVoxel::setObstacle(const Vector3i& o) {
		m_obstacle = o.x();
		m_obstacle |= o.y() << 10;
		m_obstacle |= o.z() << 20;
	}

	__host__ __device__
	inline void DistanceVoxel::setPBAUninitialised() {
		const Vector3ui temp = Vector3ui::Constant(PBA_UNINITIALISED_COORD);
		setObstacle(temp);
	}

	__host__ __device__
		inline bool DistanceVoxel::isOccupied(float col_threshold) const
	{
		//NOP
		//TODO just a stub for API compatibility
		//TODO update as soon as DistanceVoxel rework is done
		return true;
	}


	__host__ __device__
		inline void DistanceVoxel::insert(const uint32_t voxel_meaning)
	{
		//NOP
		//TODO update as soon as DistanceVoxel rework is done

		//TODO remove or hide behind #ifdef?
		printf("DistanceVoxel.insert(voxel_type: %d) should not be called! use insert(pos, type)\n", voxel_meaning);
	}

	__host__ __device__
		inline void DistanceVoxel::insert(const Vector3ui& voxel_position, const uint32_t voxel_meaning)
	{
		if (voxel_meaning == eBVM_OCCUPIED)
		{
			setObstacle(voxel_position);
			//    printf("%d/%d/%d \n",voxel_position.x,voxel_position.y,voxel_position.z);
		}
		else {
			// should not happen? print debug info
			printf("DistanceVoxel.insert: SHOULD_USE_CLEARMAP_INSTEAD_OF_INSERT(FREE)! voxel_meaning: %u\n", voxel_meaning);

			setPBAUninitialised();
		}
	}

	__host__ __device__
		inline DistanceVoxel::operator uint3() const {
		uint3 t;
		t.x = m_obstacle & 1023;
		t.y = (m_obstacle >> 10) & 1023;
		t.z = m_obstacle >> 20;
		return t;
	}

} // end of ns

#endif