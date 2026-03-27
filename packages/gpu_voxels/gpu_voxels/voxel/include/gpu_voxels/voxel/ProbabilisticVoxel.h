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
#ifndef GPU_VOXELS_VOXEL_PROBABILISTIC_VOXEL_H_INCLUDED
#define GPU_VOXELS_VOXEL_PROBABILISTIC_VOXEL_H_INCLUDED

#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {

	/**
	 * @brief Probabilistic voxel type with probability in log-odd representation
	 */
	class ProbabilisticVoxel
	{
	public:

		GVL_HOST_DEVICE
		static Probability floatToProbability(const float val)
		{
			float tmp = (std::max)((std::min)(1.0f, val), 0.0f);
			tmp = (tmp * (float(MAX_PROBABILITY) - float(MIN_PROBABILITY))) + MIN_PROBABILITY;
			return static_cast<Probability>(tmp);
		}

		GVL_HOST_DEVICE

		static float probabilityToFloat(const Probability val)
		{
			return (float(val) - float(MIN_PROBABILITY)) / (float(MAX_PROBABILITY) - float(MIN_PROBABILITY));
		}


		/**
		 * @brief ProbabilisticVoxel
		 */
		GVL_HOST_DEVICE
		ProbabilisticVoxel();

		GVL_HOST_DEVICE
		ProbabilisticVoxel(Probability p);

		GVL_HOST_DEVICE
		~ProbabilisticVoxel();

		/**
		 * @brief updateOccupancy Updates the occupancy of this voxel based on the log-odd representation.
		 * @param occupancy A new occupancy measurement.
		 * @return Returns the updated occupancy.
		 */
		GVL_HOST_DEVICE
		Probability updateOccupancy(Probability occupancy);

		/**
		 * @brief occupancy Write reference.
		 * @return
		 */
		GVL_HOST_DEVICE
		Probability& occupancy();

		/**
		 * @brief occupancy Read-only reference.
		 * @return
		 */
		GVL_HOST_DEVICE
		[[nodiscard]] const Probability& occupancy() const;

		/**
		 * @brief getOccupancy Read-only access per copy
		 * @return
		 */
		GVL_HOST_DEVICE
		[[nodiscard]] Probability getOccupancy() const;

		GVL_HOST_DEVICE
		void insert(BitVoxelMeaning voxel_meaning);

		GVL_HOST_DEVICE
		static ProbabilisticVoxel reduce(ProbabilisticVoxel voxel, ProbabilisticVoxel other_voxel);

		struct reduce_op
		{
			GVL_HOST_DEVICE
				ProbabilisticVoxel operator()(const ProbabilisticVoxel& a, const ProbabilisticVoxel& b) const
			{
				ProbabilisticVoxel tmp = a;
				tmp.updateOccupancy(b.getOccupancy());
				return tmp;
			}
		};

		GVL_HOST_DEVICE
			[[nodiscard]] bool isOccupied(float col_threshold) const;


		template<typename T>
		__host__
		friend T& operator<<(T& os, const ProbabilisticVoxel& dt)
		{
			os << static_cast<int>(dt.getOccupancy());
			return os;
		}

		__host__
		friend std::istream& operator>>(std::istream& in, ProbabilisticVoxel& dt)
		{
			Probability tmp;
			in >> tmp;
			dt.occupancy() = tmp;
			return in;
		}

	protected:
		Probability m_occupancy;
	};

} // end of ns

#endif
