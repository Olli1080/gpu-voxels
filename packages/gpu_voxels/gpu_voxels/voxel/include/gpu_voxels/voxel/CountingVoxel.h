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
 * \author  Christian Jülg
 * \date    2017-10-17
 *
 */
 //----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXEL_COUNTING_VOXEL_H_INCLUDED
#define GPU_VOXELS_VOXEL_COUNTING_VOXEL_H_INCLUDED

#include <cuda_runtime.h>
#include <cstdint>
#include <istream>

namespace gpu_voxels {

	/**
	 * @brief Counting voxel type for filtering noise data with density threshold
	 */
	class CountingVoxel
	{
	public:
		/**
		 * @brief CountingVoxel
		 */
		GVL_HOST_DEVICE
		CountingVoxel();

		GVL_HOST_DEVICE
		[[nodiscard]] bool isOccupied(uint8_t occ_threshold) const;

		GVL_HOST_DEVICE
		[[nodiscard]] int8_t getCount() const;

		GVL_HOST_DEVICE
		int8_t& count();

		GVL_HOST_DEVICE
		[[nodiscard]] const int8_t& count() const;

		GVL_HOST_DEVICE
		void insert(const uint32_t voxel_meaning);

		GVL_HOST_DEVICE
		static CountingVoxel reduce(const CountingVoxel voxel, const CountingVoxel other_voxel);

		struct reduce_op
		{
			GVL_HOST_DEVICE
			CountingVoxel operator()(const CountingVoxel& a, const CountingVoxel& b) const
			{
				CountingVoxel tmp = a;
				tmp.m_count += b.m_count;
				return tmp;
			}
		};

		template <typename T>
		GVL_HOST
		friend T& operator<<(T& os, const CountingVoxel& dt)
		{
			os << static_cast<int>(dt.m_count);
			return os;
		}

		GVL_HOST
		friend std::istream& operator>>(std::istream& in, CountingVoxel& dt)
		{
			uint8_t tmp;
			in >> tmp;
			dt.m_count = tmp;
			return in;
		}

	protected:
		int8_t m_count;
	};

} // end of ns

#endif // GPU_VOXELS_VOXEL_COUNTING_VOXEL_H_INCLUDED
