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
 * \author  Sebastian Klemm
 * \author  Florian Drews
 * \author  Christian Juelg
 * \date    2012-06-22
 *
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_CUDA_MATRICES_H_INCLUDED
#define GPU_VOXELS_CUDA_MATRICES_H_INCLUDED

#include <iostream>
#include <iomanip>
#include <cstdio>

#include <Eigen/Dense>

#include "icl_core_logging/ThreadStream.h"

namespace gpu_voxels
{
	/*
	 * Addressing within matrices is COLUMN-Major see Eigen
	 */
	
	typedef Eigen::Matrix3f Matrix3f;
	typedef Eigen::Matrix4f Matrix4f;

	/*!
	 * \brief createFromRPY Constructs a rotation matrix
	 * \param _roll
	 * \param _pitch
	 * \param _yaw
	 * \return
	 */
	__host__ __device__
	inline Eigen::Quaternionf createFromRPY(float _roll, float _pitch, float _yaw)
	{
		const Eigen::AngleAxisf roll(_roll, Eigen::Vector3f::UnitX());
		const Eigen::AngleAxisf pitch(_pitch, Eigen::Vector3f::UnitY());
		const Eigen::AngleAxisf yaw(_yaw, Eigen::Vector3f::UnitZ());

		Eigen::Quaternionf out = yaw * pitch * roll;
		return out;
	}

	/*!
	 * \brief createFromRPY Constructs a rotation matrix by first rotating around Roll, then around Pitch and finaly around Yaw.
	 * This acts in the same way as ROS TF Quaternion.setRPY().
	 * \param rpy Vector of Roll Pitch and Yaw
	 * \return Matrix where rotation is set.
	 */
	__device__ __host__
	inline Eigen::Quaternionf createFromRPY(Vector3f rpy)
	{
		return createFromRPY(rpy.x(), rpy.y(), rpy.z());
	}

	__host__ __device__
	inline [[nodiscard]] Vector3f orientationMatrixDiff(const Matrix3f& first, const Matrix3f& other)
	{
		Vector3f tmp1 = first.col(0);
		Vector3f tmp2 = other.col(0);
		Vector3f d = tmp1.cross(tmp2);

		tmp1 = first.col(1);
		tmp2 = other.col(1);
		d = d + tmp1.cross(tmp2);

		tmp1 = first.col(2);
		tmp2 = other.col(2);
		d = d + tmp1.cross(tmp2);

		return { asinf(d.x() / 2.f), asinf(d.y() / 2.f), asinf(d.z() / 2.f) };
	}

	__host__
	inline std::ostream& operator<<(std::ostream& out, const Matrix3f& matrix)
	{
		out.precision(3);
		out << "\n" << std::fixed <<
			"[" << std::setw(10) << matrix(0, 0) << ", " << std::setw(10) << matrix(0, 1) << ", " << std::setw(10) << matrix(0, 2) << ",\n"
			" " << std::setw(10) << matrix(1, 0) << ", " << std::setw(10) << matrix(1, 1) << ", " << std::setw(10) << matrix(1, 2) << ",\n"
			" " << std::setw(10) << matrix(2, 0) << ", " << std::setw(10) << matrix(2, 1) << ", " << std::setw(10) << matrix(2, 2) << "]"
			<< std::endl;
		return out;
	}

	__host__
	inline icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Matrix3f& matrix)
	{
		out << "\n" <<
			"[" << matrix(0, 0) << ", " << matrix(0, 1) << ", " << matrix(0, 2) << ",\n"
			" " << matrix(1, 0) << ", " << matrix(1, 1) << ", " << matrix(1, 2) << ",\n"
			" " << matrix(2, 0) << ", " << matrix(2, 1) << ", " << matrix(2, 2) << "]"
			<< icl_core::logging::endl;
		return out;
	}

	__device__ __host__
	inline void print(const Eigen::Matrix4f& mat)
	{
		printf("  %0.7f  %0.7f  %0.7f  %0.7f\n", mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3));
		printf("  %0.7f  %0.7f  %0.7f  %0.7f\n", mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3));
		printf("  %0.7f  %0.7f  %0.7f  %0.7f\n", mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3));
		printf("  %0.7f  %0.7f  %0.7f  %0.7f\n\n", mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3));
	}

	__host__
	inline std::ostream& operator<<(std::ostream& out, const Matrix4f& matrix)
	{
		out.precision(3);
		out << "\n" << std::fixed <<
			"[" << std::setw(10) << matrix(0, 0) << ", " << std::setw(10) << matrix(0, 1) << ", " << std::setw(10) << matrix(0, 2) << ", " << std::setw(10) << matrix(0, 3) << ",\n"
			" " << std::setw(10) << matrix(1, 0) << ", " << std::setw(10) << matrix(1, 1) << ", " << std::setw(10) << matrix(1, 2) << ", " << std::setw(10) << matrix(1, 3) << ",\n"
			" " << std::setw(10) << matrix(2, 0) << ", " << std::setw(10) << matrix(2, 1) << ", " << std::setw(10) << matrix(2, 2) << ", " << std::setw(10) << matrix(2, 3) << ",\n"
			" " << std::setw(10) << matrix(3, 0) << ", " << std::setw(10) << matrix(3, 1) << ", " << std::setw(10) << matrix(3, 2) << ", " << std::setw(10) << matrix(3, 3) << "]"
			<< std::endl;
		return out;
	}

	__host__
	inline icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Matrix4f& matrix)
	{
		out << "\n" <<
			"[" << matrix(0, 0) << ", " << matrix(0, 1) << ", " << matrix(0, 2) << ", " << matrix(0, 3) << ",\n"
			" " << matrix(1, 0) << ", " << matrix(1, 1) << ", " << matrix(1, 2) << ", " << matrix(1, 3) << ",\n"
			" " << matrix(2, 0) << ", " << matrix(2, 1) << ", " << matrix(2, 2) << ", " << matrix(2, 3) << ",\n"
			" " << matrix(3, 0) << ", " << matrix(3, 1) << ", " << matrix(3, 2) << ", " << matrix(3, 3) << "]"
			<< icl_core::logging::endl;
		return out;
	}
} // end of namespace
#endif