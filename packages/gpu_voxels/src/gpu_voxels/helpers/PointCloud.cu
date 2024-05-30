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
 * \author  Herbert Pietrzyk
 * \date    2016-05-25
 *
 */
//----------------------------------------------------------------------
#include "PointCloud.h"
#include <gpu_voxels/helpers/kernels/MetaPointCloudOperations.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>

#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace gpu_voxels
{
    PointCloud::PointCloud(const std::vector<Vector3f>& points)
    {
         m_points_dev = std::make_unique<ThrustDeviceVector<Vector3f>>(points.begin(), points.end());
    }

    PointCloud::PointCloud(const Vector3f* points, uint32_t size)
    {
    	m_points_dev = std::make_unique<ThrustDeviceVector<Vector3f>>(points, points + size);
    }

    PointCloud::PointCloud(const PointCloud& other)
    {
        m_points_dev = std::make_unique<ThrustDeviceVector<Vector3f>>(other.getPointsDevice());
    }

    PointCloud::PointCloud(const std::string& path_to_file, bool use_model_path)
    {
        std::vector<Vector3f> host_point_cloud;
        if (!file_handling::PointcloudFileHandler::Instance()->loadPointCloud(path_to_file, use_model_path, host_point_cloud))
        {
            LOGGING_ERROR_C(Gpu_voxels_helpers, PointCloud,
                "Could not read file " << path_to_file << icl_core::logging::endl);
            return;
        }
        
        m_points_dev = std::make_unique<ThrustDeviceVector<Vector3f>>(host_point_cloud.begin(), host_point_cloud.end());
    }


    PointCloud& PointCloud::operator=(const PointCloud& other)
    {
        if (this != &other) // self-assignment check expected
            m_points_dev = std::make_unique<ThrustDeviceVector<Vector3f>>(other.getPointsDevice());
        
        return *this;
    }

    void PointCloud::resize(uint32_t new_number_of_points)
    {
        m_points_dev->resize(new_number_of_points);
    }

    bool PointCloud::operator==(const PointCloud& other) const
    {
        // Things are clear if self comparison:
        if (this == &other)
        {
            LOGGING_DEBUG_C(Gpu_voxels_helpers, PointCloud, "Clouds are the same object." << icl_core::logging::endl);
            return true;
        }
        // Size has to match:
        if (m_points_dev->size() != other.m_points_dev->size())
        {
            LOGGING_DEBUG_C(Gpu_voxels_helpers, PointCloud, "Pointcloud size does not match." << icl_core::logging::endl);
            return false;
        }

        return thrust::equal(thrust::device, m_points_dev->begin(), m_points_dev->end(), other.m_points_dev->begin());
    }

    void PointCloud::add(const PointCloud& cloud)
    {
        this->add(cloud.getPoints());
    }

    void PointCloud::add(const std::vector<Vector3f>& points)
    {
        m_points_dev->reserve(m_points_dev->size() + points.size());
        m_points_dev->insert(m_points_dev->end(), points.begin(), points.end());
    }

    void PointCloud::add(const thrust::host_vector<Vector3f>& points)
    {
        m_points_dev->reserve(m_points_dev->size() + points.size());
        m_points_dev->insert(m_points_dev->end(), points.begin(), points.end());
    }

    void PointCloud::update(const PointCloud& cloud)
    {
        this->update(cloud.getPoints());
    }

    void PointCloud::update(const std::vector<Vector3f>& points)
    {
        m_points_dev = std::make_unique<ThrustDeviceVector<Vector3f>>(points.begin(), points.end());
    }

    void PointCloud::update(const thrust::host_vector<Vector3f>& points)
    {
        m_points_dev = std::make_unique<ThrustDeviceVector<Vector3f>>(points);
    }

    void PointCloud::transformSelf(const Matrix4f& transform)
    {
        this->transform(transform, *this);
    }

    void PointCloud::transform(const Matrix4f& transform, PointCloud& transformed_cloud) const
    {
        if (&transformed_cloud != this)
            transformed_cloud.resize(static_cast<uint32_t>(m_points_dev->size()));
        
        // transform the cloud via Kernel.
        thrust::transform(thrust::cuda::par_nosync, m_points_dev->begin(), m_points_dev->end(), 
            transformed_cloud.getPointsDevice().begin(), KernelTransform(transform));
        CHECK_CUDA_ERROR();

        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void PointCloud::scaleSelf(const Vector3f& scaling)
    {
        this->scale(scaling, *this);
    }

    void PointCloud::scale(const Vector3f& scaling, PointCloud& scaled_cloud) const
    {
	    if (&scaled_cloud != this)
            scaled_cloud.resize(static_cast<uint32_t>(m_points_dev->size()));

        // transform the cloud via Kernel.
        thrust::transform(thrust::cuda::par_nosync, m_points_dev->begin(), m_points_dev->end(),
            scaled_cloud.getPointsDevice().begin(), KernelScale(scaling));
        CHECK_CUDA_ERROR();

        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    }

    thrust::device_vector<Vector3f>& PointCloud::getPointsDevice()
    {
        return *m_points_dev;
    }

    const thrust::device_vector<Vector3f>& PointCloud::getPointsDevice() const
    {
        return *m_points_dev;
    }

    thrust::host_vector<Vector3f> PointCloud::getPoints() const
    {
        thrust::host_vector<Vector3f> out = *m_points_dev;
        return out;
    }

    uint32_t PointCloud::getPointCloudSize() const
    {
        return static_cast<uint32_t>(m_points_dev->size());
    }

    void PointCloud::print() const
    {
        int i = 0;
        for (const auto& point : this->getPoints())
            std::cout << "Point " << i++ << ": (" << point.x() << ", " << point.y() << ", " << point.z() << ")" << std::endl;

        std::cout << std::endl << std::endl;
    }
}// end namespace gpu_voxels