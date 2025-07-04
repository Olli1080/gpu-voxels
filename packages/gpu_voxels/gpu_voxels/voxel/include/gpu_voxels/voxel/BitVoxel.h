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
 * \date    2014-07-08
 *
 */
 //----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXEL_BIT_VOXEL_H_INCLUDED
#define GPU_VOXELS_VOXEL_BIT_VOXEL_H_INCLUDED

#include <gpu_voxels/helpers/BitVector.h>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {

	/**
	 * @brief Voxel which represents a bit vector.
	 */
	template<std::size_t length>
	class BitVoxel
	{
	public:

		__host__ __device__
		BitVoxel();

		__host__ __device__
		bool operator==(const BitVoxel& other) const;

		__host__ __device__
		BitVector<length>& bitVector();

		__host__ __device__
		const BitVector<length>& bitVector() const;

		__host__ __device__
		void insert(const BitVoxelMeaning voxel_meaning);

		__host__ __device__
		static BitVoxel reduce(const BitVoxel voxel, const BitVoxel other_voxel);

		struct reduce_op
		{
			__host__ __device__
			BitVoxel operator()(const BitVoxel& a, const BitVoxel& b) const;
		};

		__host__ __device__
		[[nodiscard]] bool isOccupied(float col_threshold) const;

		/**
		 * @brief operator >> Overloaded ostream operator. Please note that the output bit string is starting from
		 * Type 0.
		 */
		template<typename T>
		__host__
		friend T& operator<<(T& os, const BitVoxel& dt)
		{
			os << dt.bitVector();
			return os;
		}

		/**
		 * @brief operator << Overloaded istream operator. Please note that the input bit string should
		 * be starting from Type 0 and it should be complete, meaning it should have all Bits defined.
		 */
		__host__
		friend std::istream& operator>>(std::istream& in, BitVoxel& dt)
		{
			in >> dt.bitVector();
			return in;
		}

	protected:
		BitVector<length> m_bit_vector;
	};

} // end of ns

#endif
