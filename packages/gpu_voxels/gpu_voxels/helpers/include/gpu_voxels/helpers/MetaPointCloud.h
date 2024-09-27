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
 * \date    2014-06-17
 *
 * This is a management structure to handle arrays of PointClouds
 * on the GPU. Such as RobotLinks or sensor-data.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_METAPOINTCLOUD_H_INCLUDED
#define GPU_VOXELS_HELPERS_METAPOINTCLOUD_H_INCLUDED

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include <thrust/device_vector.h>

#include <gpu_voxels/helpers/cuda_datatypes.hpp>

namespace gpu_voxels
{
    class PointCloud;
    class MetaPointCloud
    {
    public:

        MetaPointCloud();
        explicit MetaPointCloud(const std::vector<std::string>& _point_cloud_files, const std::filesystem::path& model_path);
        explicit MetaPointCloud(const std::vector<std::string>& _point_cloud_files,
            const std::vector<std::string>& _point_cloud_names, const std::filesystem::path& model_path);
        explicit MetaPointCloud(const std::vector<uint32_t>& _point_cloud_sizes);
        explicit MetaPointCloud(const std::vector< std::vector<Vector3f> >& point_clouds);

        // Deep Copy Operators
        explicit MetaPointCloud(const MetaPointCloud& other);
        MetaPointCloud& operator=(const MetaPointCloud& other);

        // Deep equality check
        bool operator==(const MetaPointCloud& other) const;

        //! Destructor
        ~MetaPointCloud();

        void addCloud(uint32_t cloud_size);
        void addCloud(const std::vector<Vector3f>& cloud, bool sync = false, const std::string& name = "");
        
        void addCloud(const PointCloud& cloud, const std::string& name = "");
        void addClouds(const std::vector<std::string>& _point_cloud_files, const std::filesystem::path& model_path);
        [[nodiscard]] std::string getCloudName(uint16_t i) const;
        [[nodiscard]] const std::map<uint16_t, std::string>& getCloudNames() const;
        [[nodiscard]] std::optional<uint16_t> getCloudNumber(const std::string& name) const;
        [[nodiscard]] bool hasCloud(const std::string& name) const;


        /*!
         * \brief syncToDevice copies a specific map to the GPU
         * \param cloud The cloud that will be synced
         */
        void syncToDevice(uint16_t cloud);

        /*!
         * \brief syncToDevice Syncs all clouds to the device at once.
         */
        void syncToDevice();

        void syncToHost();

        /*!
         * \brief updatePointCloud This updates a specific cloud on the host.
         * Call syncToDevice() after updating all clouds or set sync to true
         * to only sync this current cloud to the GPU.
         * \param cloud Id of the cloud to update
         * \param pointcloud The new cloud. May differ in size.
         * \param sync If set to true, only this modified cloud is synced to the GPU.
         */
        void updatePointCloud(uint16_t cloud, const std::vector<Vector3f>& pointcloud, bool sync = false);

        /*!
         * \brief updatePointCloud This updates a specific cloud on the host.
         * Call syncToDevice() after updating all clouds or set sync to true
         * to only sync this current cloud to the GPU.
         * \param cloud Id of the cloud to update
         * \param pointcloud The new cloud. May differ in size.
         */
        void updatePointCloud(uint16_t cloud, const PointCloud& pointcloud);

        void updatePointCloud(uint16_t cloud, const thrust::device_vector<Vector3f>& pointcloud);

        /*!
         * \brief updatePointCloud This updates a specific cloud on the host.
         * Call syncToDevice() after updating all clouds or set sync to true
         * to only sync this current cloud to the GPU.
         * \param cloud_name Name of the cloud to update
         * \param pointcloud The new cloud. May differ in size.
         * \param sync If set to true, only this modified cloud is synced to the GPU.
         */
        void updatePointCloud(const std::string& cloud_name, const std::vector<Vector3f>& pointcloud, bool sync = false);

        /*!
         * \brief updatePointCloud This updates a specific cloud on the host.
         * Call syncToDevice() after updating all clouds or set sync to true
         * to only sync this current cloud to the GPU.
         * \param cloud Id of the cloud to update
         * \param pointcloud The new cloud
         * \param pointcloud_size Size of the new cloud
         * \param sync If set to true, only this modified cloud is synced to the GPU.
         */
        void updatePointCloud(uint16_t cloud, const Vector3f* pointcloud, uint32_t pointcloud_size, bool sync = false);

        void updatePointCloud(uint16_t cloud, const std::span<Vector3f>& pointcloud, uint32_t pointcloud_size, bool sync = false);

        /*!
         * \brief getNumberOfPointclouds
         * \return Number of clouds in the MetaPointCloud
         */
        [[nodiscard]] uint16_t getNumberOfPointclouds() const;

        /*!
         * \brief getPointcloudSize Returns the number of elements in one point cloud
         * \param cloud The ID of the cloud.
         * \return Numper of points in one pointcloud.
         */
        [[nodiscard]] uint32_t getPointCloudSize(uint16_t cloud = 0) const;

        /*!
         * \brief getAccumulatedPointcloudSize
         * \return Accumulated size of all point clouds
         */
        [[nodiscard]] uint32_t getAccumulatedPointcloudSize() const;

        [[nodiscard]] bool empty() const;

        /*!
         * \brief getPointcloudSizes
         * \return A vector of the sizes of all point clouds.
         */
        [[nodiscard]] const thrust::host_vector<uint32_t>& getPointcloudSizes() const;

        /*!
         * \brief getPointCloud
         * \param cloud Which cloud to return.
         * \return A pointer to the host point cloud.
         */
        [[nodiscard]] std::span<Vector3f> getPointCloud(uint16_t cloud) const;

        /*!
         * \brief getDevicePointer
         * \return Returns a writable pointer to the device data
         */
        [[nodiscard]] thrust::device_ptr<MetaPointCloudStruct> getDevicePointer() const;

        /*!
         * \brief getDeviceConstPointer
         * \return Returns a const pointer to the device data for RO access
         */
        [[nodiscard]] thrust::device_ptr<const MetaPointCloudStruct> getDeviceConstPointer() const;

        void debugPointCloud() const;


        /*!
         * \brief transform transforms this whole MetaPointCloud and writes it into the output MetaPointCloud.
         * \param transformation The transformation to apply
         * \param transformed_cloud The transformed cloud. Has to be of the same size as this cloud!
         */
        void transform(const Matrix4f& transformation, MetaPointCloud& transformed_cloud) const;


        /*!
         * \brief transform transforms a subcloud of this MetaPointCloud and writes it into the output MetaPointCloud.
         * \param subcloud_to_transform The ID of the subcloud which is transformed
         * \param transformation The transformation to apply
         * \param transformed_cloud The transformed cloud. Has to be of the same size as this cloud!
         */
        void transformSubCloud(uint16_t subcloud_to_transform, const Matrix4f& transformation, const MetaPointCloud& transformed_cloud) const;

        void transformSubClouds(uint16_t startcloud_to_transform, const std::vector<Matrix4f>& transformations, const MetaPointCloud& transformed_cloud) const;

        //tuple of <subcloud_to_transform, transformation>
        void transformSubClouds(const std::list<std::tuple<uint16_t, Matrix4f>>& subcloud_transforms, const MetaPointCloud& transformed_cloud) const;

        /*!
         * \brief transform transforms this whole MetaPointCloud
         * \param transformation The transformation to apply
         */
        void transformSelf(const Matrix4f& transformation);


        /*!
         * \brief transform transforms a subcloud of this MetaPointCloud
         * \param subcloud_to_transform The ID of the subcloud which is transformed
         * \param transformation The transformation to apply
         */
        void transformSelfSubCloud(uint8_t subcloud_to_transform, const Matrix4f& transformation) const;

    private:

        void addCloud(const Vector3f* points, uint32_t pointcloud_size, bool sync = false, const std::string& name = "");
        void addCloud(const thrust::device_vector<Vector3f>& point_cloud, const std::string& name = "");

        /*!
         * \brief Init does the allocation of Device and Host memory
         * \param _point_cloud_sizes The point cloud sizes that are required for the malloc
         */
        void init(const thrust::host_vector<uint32_t>& _point_cloud_sizes);

        /*!
         * \brief MetaPointCloud::Destruct Private destructor that is also called, when a
         * new cloud is added.
         */
        void destruct();
        
        std::map<uint16_t, std::string> m_point_cloud_names;
        
        std::vector<Vector3f> m_accumulated_cloud;

        std::shared_ptr<MetaPointCloudStructLocal> m_point_clouds_local;
        std::shared_ptr<MetaPointCloudStruct> m_dev_point_clouds_local;

        thrust::device_vector<Vector3f> m_dev_ptr_to_accumulated_cloud;
        thrust::device_ptr<MetaPointCloudStruct> m_dev_ptr_to_point_clouds_struct;
        std::vector<thrust::device_ptr<Vector3f>> m_dev_ptrs_to_addrs;
        thrust::device_vector<uint32_t> m_dev_ptr_to_cloud_sizes;
        thrust::device_vector<Vector3f*> m_dev_ptr_to_clouds_base_addresses;
    };
}
#endif