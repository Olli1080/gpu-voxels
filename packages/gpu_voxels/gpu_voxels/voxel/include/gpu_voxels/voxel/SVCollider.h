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
* \date    2015-6-16
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VOXEL_SVCOLLIDER_H
#define GPU_VOXELS_VOXEL_SVCOLLIDER_H

#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>
#include <gpu_voxels/voxel/DistanceVoxel.h>

namespace gpu_voxels {


	class SVCollider
	{
	public:

		GVL_HOST_DEVICE
		SVCollider();

		GVL_HOST_DEVICE
		SVCollider(const float coll_threshold, const size_t window_size = 0);

		GVL_HOST_DEVICE
		SVCollider(const Probability threshold1, const Probability threshold2, const size_t window_size = 0);

		GVL_HOST_DEVICE
		[[nodiscard]] bool collide(const ProbabilisticVoxel& v1, const ProbabilisticVoxel& v2) const;

		GVL_HOST_DEVICE
		[[nodiscard]] bool collide(const ProbabilisticVoxel& v1) const;

		template<std::size_t length>
		GVL_HOST_DEVICE
		bool collide(const ProbabilisticVoxel& v1, const BitVoxel<length>& v2) const;

		template<std::size_t length>
		GVL_HOST_DEVICE
		bool collide(const BitVoxel<length>& v1, const ProbabilisticVoxel& v2) const;

		//template<std::size_t length>
		//GVL_HOST_DEVICE
		//bool collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2) const;

		template<std::size_t length>
		GVL_HOST_DEVICE
		bool collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2, BitVector<length>* collisions, const uint32_t sv_offset) const;

		template<std::size_t length>
		GVL_HOST_DEVICE
		bool collide(const BitVoxel<length>& v1, const ProbabilisticVoxel& v2, BitVector<length>* collisions, const uint32_t sv_offset = 0) const;

		template<std::size_t length>
		GVL_HOST_DEVICE
		bool collide(const ProbabilisticVoxel& v1, const BitVoxel<length>& v2, BitVector<length>* collisions, const uint32_t sv_offset = 0) const;

		template<class OtherVoxel>
		GVL_HOST_DEVICE
		bool collide(const DistanceVoxel& v1, const OtherVoxel& v2) const;

		template<class OtherVoxel>
		GVL_HOST_DEVICE
		bool collide(const OtherVoxel& v1, const DistanceVoxel& v2) const;

	protected:

		GVL_HOST_DEVICE
		static Probability floatToProbability(const float val);

	protected:
		Probability m_threshold1, m_threshold2;
		size_t m_type_range;
	};

} // end of ns

#endif // GPU_VOXELS_VOXELMAP_SVCOLLIDER_H
