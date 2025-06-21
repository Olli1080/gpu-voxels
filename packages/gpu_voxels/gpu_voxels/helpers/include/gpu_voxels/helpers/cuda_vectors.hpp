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
#ifndef GPU_VOXELS_CUDA_VECTORS_H_INCLUDED
#define GPU_VOXELS_CUDA_VECTORS_H_INCLUDED

#include <Eigen/Eigen>

#include "icl_core_logging/ThreadStream.h"

namespace gpu_voxels {
    
    typedef Eigen::Vector<uint32_t, 3> Vector3ui;
    typedef Eigen::Vector<int32_t, 3> Vector3i;
    typedef Eigen::Vector<float, 3> Vector3f;

    typedef Eigen::Vector<int32_t, 4> Vector4i;
    typedef Eigen::Vector<float, 4> Vector4f;
    
    __device__ __host__
    inline uint3 convert(Vector3ui val)
    {
        return { val.x(), val.y(), val.z() };
    }

    template<typename T>
    __host__
    inline std::ostream& operator<<(std::ostream& out, const Eigen::Vector<T, 3>& vector)
    {
        out << "(x, y, z) = (" << vector.x() << ", " << vector.y() << ", " << vector.z() << ")" << std::endl;
        return out;
    }

    template<typename T>
    __host__
    inline icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Eigen::Vector<T, 3>& vector)
    {
        out << "(x, y, z) = (" << vector.x() << ", " << vector.y() << ", " << vector.z() << ")" << icl_core::logging::endl;
        return out;
    }

    template<typename T>
    __host__
    std::ostream& operator<<(std::ostream& out, const Eigen::Vector<T, 4>& vector)
    {
        out << "(x, y, z, w) = (" << vector.x() << ", " << vector.y() << ", " << vector.z() << ", " << vector.w() << ")" << std::endl;
        return out;
    }

    template<typename T>
    __host__
    inline icl_core::logging::ThreadStream& operator<<(icl_core::logging::ThreadStream& out, const Eigen::Vector<T, 4>& vector)
    {
        out << "(x, y, z, w) = (" << vector.x() << ", " << vector.y() << ", " << vector.z() << ", " << vector.w() << ")" << icl_core::logging::endl;
        return out;
    }

    template<typename T>
    inline Eigen::Vector<T, 3> operator%(const Eigen::Vector<T, 3>& a, const Eigen::Vector<T, 3>& b)
    {
        Eigen::Vector<T, 3> result;
        result.x() = a.x() % b.x();
        result.y() = a.y() % b.y();
        result.z() = a.z() % b.z();
        return result;
    }

    template<typename T>
    __device__ __host__
	inline Eigen::Vector<T, 3> operator/(const Eigen::Vector<T, 3>& a, const Eigen::Vector<T, 3>& b)
    {
        Eigen::Vector<T, 3> result;
        result.x() = a.x() / b.x();
        result.y() = a.y() / b.y();
        result.z() = a.z() / b.z();
        return result;
    }
    
    __device__ __host__
    __forceinline__ bool operator<=(const Vector3ui& a, const Vector3ui& b)
    {
        return a.x() <= b.x() && a.y() <= b.y() && a.z() <= b.z();
    }

    __device__ __host__
    __forceinline__ bool operator>=(const Vector3ui& a, const Vector3ui& b)
    {
        return a.x() >= b.x() && a.y() >= b.y() && a.z() >= b.z();
    }
    
    __device__ __host__
    __forceinline__ bool operator<(const Vector3ui& a, const Vector3ui& b)
    {
        return a.x() < b.x() && a.y() < b.y() && a.z() < b.z();
    }

    __device__ __host__
    __forceinline__ bool operator>(const Vector3ui& a, const Vector3ui& b)
    {
        return a.x() > b.x() && a.y() > b.y() && a.z() > b.z();
    }
    
    __device__ __host__
    __forceinline__ Vector3ui operator>>(const Vector3ui& a, const uint32_t shift)
    {
        return {a.x() >> shift, a.y() >> shift, a.z() >> shift};
    }

    __device__ __host__
        __forceinline__ Vector3ui operator<<(const Vector3ui& a, const uint32_t shift)
    {
        return {a.x() << shift, a.y() << shift, a.z() << shift};
    }

    __device__ __host__
    __forceinline__ Vector3ui operator&(const Vector3ui& a, const uint32_t value)
    {
        return {a.x() & value, a.y() & value, a.z() & value};
    }
} // end of namespace
#endif