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
 * \date    2015-01-05
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/primitive_array/PrimitiveArray.h>
#include <gpu_voxels/helpers/cuda_handling.hpp>

namespace gpu_voxels {
    namespace primitive_array {

        PrimitiveArray::PrimitiveArray(Vector3ui dim, float voxel_side_length, PrimitiveType type) :
            m_dim(std::move(dim)),
            m_voxel_side_length(voxel_side_length),
            m_primitive_type(type),
            m_num_entities(0),
            m_dev_ptr_to_primitive_positions(nullptr)
        {
        }

        PrimitiveArray::~PrimitiveArray()
        {
            if (m_dev_ptr_to_primitive_positions)
            {
                //delete old array
                cudaFree(m_dev_ptr_to_primitive_positions);
                m_dev_ptr_to_primitive_positions = nullptr;
            }
        }

        // metric coords
        void PrimitiveArray::setPoints(const std::vector<Vector4f>& metricPoints)
        {
            std::vector<Vector4f> voxelPoints(metricPoints.size());
            //calculate voxel coordinates
            for (size_t i = 0; i < metricPoints.size(); i++)
            {
                voxelPoints[i] = Vector4f(floor(metricPoints[i].x() / m_voxel_side_length) + 0.5f,
                    floor(metricPoints[i].y() / m_voxel_side_length) + 0.5f,
                    floor(metricPoints[i].z() / m_voxel_side_length) + 0.5f,
                    (metricPoints[i].w() / m_voxel_side_length) / 2.f);
            }
            storePoints(voxelPoints);
        }

        // metric coords
        void PrimitiveArray::setPoints(const std::vector<Vector3f>& metricPoints, float diameter)
        {
            std::vector<Vector4f> voxelPoints(metricPoints.size());
            for (size_t i = 0; i < metricPoints.size(); i++)
            {
                voxelPoints[i] = Vector4f(floor(metricPoints[i].x() / m_voxel_side_length) + 0.5f,
                    floor(metricPoints[i].y() / m_voxel_side_length) + 0.5f,
                    floor(metricPoints[i].z() / m_voxel_side_length) + 0.5f,
                    (diameter / m_voxel_side_length) / 2.f);
            }
            storePoints(voxelPoints);
        }

        // voxel coords
        void PrimitiveArray::setPoints(const std::vector<Vector4i>& points)
        {
            std::vector<Vector4f> points_f(points.size());
            for (size_t i = 0; i < points.size(); i++)
            {
                points_f[i] = Vector4f(points[i].x() + 0.5f,
                    points[i].y() + 0.5f,
                    points[i].z() + 0.5f,
                    points[i].w() / 2.f);
            }
            storePoints(points_f);
        }

        // voxel coords
        void PrimitiveArray::setPoints(const std::vector<Vector3i>& points, uint32_t diameter)
        {
            std::vector<Vector4f> points_4d(points.size());
            for (size_t i = 0; i < points.size(); i++)
            {
                points_4d[i] = Vector4f(points[i].x() + 0.5f,
                    points[i].y() + 0.5f,
                    points[i].z() + 0.5f,
                    diameter / 2.f);
            }
            storePoints(points_4d);
        }

        void PrimitiveArray::storePoints(const std::vector<Vector4f>& points)
        {
            // if only the poses have changed, but the number of primitives stays constant,
            // we only have to copy over the new data. Else we need to malloc new mem!
            if (points.size() != m_num_entities)
            {
                if (m_dev_ptr_to_primitive_positions)
                {
                    //delete old array
                    cudaFree(m_dev_ptr_to_primitive_positions);
                    m_dev_ptr_to_primitive_positions = nullptr;
                }
                // allocate the accumulated memory for the positions of the primitives
                m_num_entities = static_cast<uint32_t>(points.size());
                HANDLE_CUDA_ERROR(
                    cudaMalloc(reinterpret_cast<void**>(&m_dev_ptr_to_primitive_positions), m_num_entities * sizeof(Vector4f)));
            }

            HANDLE_CUDA_ERROR(
                cudaMemcpy(m_dev_ptr_to_primitive_positions, points.data(), m_num_entities * sizeof(Vector4f),
                    cudaMemcpyHostToDevice));
        }

        std::size_t PrimitiveArray::getMemoryUsage()
        {
            return m_num_entities * sizeof(Vector4f);
        }

    } // end of namespace
} // end of namespace