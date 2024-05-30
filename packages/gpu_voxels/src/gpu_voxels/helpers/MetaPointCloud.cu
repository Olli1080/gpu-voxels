// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \date    2014-06-17
 *
 */
//----------------------------------------------------------------------
#include "MetaPointCloud.h"

#include <numeric>
#include <ranges>

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>

#include <gpu_voxels/helpers/PointCloud.h>

#include <gpu_voxels/helpers/kernels/MetaPointCloudOperations.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>

#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/cuda_handling.h>

namespace gpu_voxels
{
    struct MetaPointCloud::CUDA_impl
    {
        thrust::device_vector<Vector3f> m_dev_ptr_to_accumulated_cloud;
        thrust::device_ptr<MetaPointCloudStruct> m_dev_ptr_to_point_clouds_struct;
        thrust::device_vector<uint32_t> m_dev_ptr_to_cloud_sizes;
        thrust::device_vector<Vector3f*> m_dev_ptr_to_clouds_base_addresses;
    };

    void MetaPointCloud::init(const thrust::host_vector<uint32_t>& _point_cloud_sizes)
    {
        const auto num_clouds = static_cast<uint16_t>(_point_cloud_sizes.size());
        
        // allocate point clouds space on host:
        m_point_clouds_local = std::make_shared<MetaPointCloudStructLocal>();
        m_point_clouds_local->num_clouds = num_clouds;
        m_point_clouds_local->cloud_sizes.reserve(num_clouds);
        m_point_clouds_local->clouds_base_addresses.reserve(num_clouds);

        auto accumulated_pointcloud_size = std::accumulate(_point_cloud_sizes.begin(), _point_cloud_sizes.end(), 0u);

        // Memory is only allocated once with the accumulated cloud size:
        m_point_clouds_local->accumulated_cloud_size = accumulated_pointcloud_size;
        m_accumulated_cloud = std::vector<Vector3f>(accumulated_pointcloud_size, Vector3f::Zero());

        // The pointers in clouds_base_addresses point into the accumulated memory:
        auto tmp_addr = m_accumulated_cloud.begin();
        for (uint16_t i = 0; i < num_clouds; ++i)
        {
            m_point_clouds_local->cloud_sizes.push_back(_point_cloud_sizes[i]);
            m_point_clouds_local->clouds_base_addresses.push_back(tmp_addr);
            tmp_addr += _point_cloud_sizes[i];
        }

        // allocate structure on device
        cuda_impl_->m_dev_ptr_to_point_clouds_struct = thrust::device_malloc<MetaPointCloudStruct>(1);

        // allocate space for array of point clouds sizes on device and save pointers in host local copy:
        cuda_impl_->m_dev_ptr_to_cloud_sizes = m_point_clouds_local->cloud_sizes;

        // allocate space for array of point clouds base addresses on device and save pointers in host local copy:
        cuda_impl_->m_dev_ptr_to_clouds_base_addresses.resize(num_clouds);

        // allocate the accumulated point cloud space
        cuda_impl_->m_dev_ptr_to_accumulated_cloud.resize(accumulated_pointcloud_size);

        // copy the base addresses to the device
        m_dev_ptrs_to_addrs.resize(num_clouds);
        auto ptr_iterator = cuda_impl_->m_dev_ptr_to_accumulated_cloud.data();
        for (uint16_t i = 0; i < num_clouds; i++)
        {
            m_dev_ptrs_to_addrs[i] = ptr_iterator;
            //printf("Addr of cloud %d = %p\n", i , m_dev_ptrs_to_addrs[i]);
            ptr_iterator += _point_cloud_sizes[i];
        }
        HANDLE_CUDA_ERROR(
            cudaMemcpy(cuda_impl_->m_dev_ptr_to_clouds_base_addresses.data().get(), m_dev_ptrs_to_addrs.data(), num_clouds * sizeof(Vector3f*),
                cudaMemcpyHostToDevice));

        //printf("Addr of m_dev_ptr_to_clouds_base_addresses: %p\n", m_dev_ptr_to_clouds_base_addresses);

        // copy the structure with the device pointers to the device
        m_dev_point_clouds_local = std::make_shared<MetaPointCloudStruct>();
        m_dev_point_clouds_local->num_clouds = num_clouds;
        m_dev_point_clouds_local->accumulated_cloud_size = accumulated_pointcloud_size;
        m_dev_point_clouds_local->cloud_sizes = cuda_impl_->m_dev_ptr_to_cloud_sizes.data().get();
        m_dev_point_clouds_local->clouds_base_addresses = cuda_impl_->m_dev_ptr_to_clouds_base_addresses.data().get();
        HANDLE_CUDA_ERROR(
            cudaMemcpy(cuda_impl_->m_dev_ptr_to_point_clouds_struct.get(), m_dev_point_clouds_local.get(), sizeof(MetaPointCloudStruct),
                cudaMemcpyHostToDevice));

        //  LOGGING_DEBUG_C(
        //      Gpu_voxels_helpers,
        //      MetaPointCloud,
        //      "This MetaPointCloud requires: " << (m_accumulated_pointcloud_size * sizeof(Vector3f)) * cBYTE2MBYTE << "MB on the GPU and on the Host" << endl);

    }


    void MetaPointCloud::addClouds(const std::vector<std::string>& _point_cloud_files, bool use_model_path)
    {
        auto point_clouds = std::vector<std::vector<Vector3f>>();
        point_clouds.reserve(_point_cloud_files.size());

        for (const auto& _point_cloud_file : _point_cloud_files)
        {
            std::vector<Vector3f> tempCloud;
            if (!file_handling::PointcloudFileHandler::Instance()->loadPointCloud(_point_cloud_file, use_model_path, tempCloud))
            {
                LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                    "Could not read file " << _point_cloud_file << icl_core::logging::endl);
                return;
            }
            point_clouds.emplace_back(std::move(tempCloud));
        }

        std::vector<uint32_t> point_cloud_sizes;
        point_cloud_sizes.reserve(point_clouds.size());

        for (const auto& point_cloud : point_clouds)
            point_cloud_sizes.emplace_back(static_cast<uint32_t>(point_cloud.size()));

        init(point_cloud_sizes);

        for (uint16_t i = 0; i < static_cast<uint16_t>(point_clouds.size()); ++i)
            updatePointCloud(i, point_clouds.at(i), false);

        syncToDevice();
    }

    MetaPointCloud::MetaPointCloud(const std::vector<std::string>& _point_cloud_files, bool use_model_path)
    {
        addClouds(_point_cloud_files, use_model_path);
    }

    MetaPointCloud::MetaPointCloud(const std::vector<std::string>& _point_cloud_files,
        const std::vector<std::string>& _point_cloud_names, bool use_model_path)

    {
        addClouds(_point_cloud_files, use_model_path);

        if (_point_cloud_files.size() == _point_cloud_names.size())
        {
            for (uint16_t i = 0; i < static_cast<uint16_t>(_point_cloud_files.size()); i++)
                m_point_cloud_names[i] = _point_cloud_names[i];
        }
        else {
            LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                "Number of names differs to number of pointcloud files!" << icl_core::logging::endl);
        }
    }


    MetaPointCloud::MetaPointCloud()
	    : cuda_impl_(std::make_unique<CUDA_impl>())
    {
        init({});
    }

    MetaPointCloud::MetaPointCloud(const std::vector<uint32_t>& _point_cloud_sizes)
        : cuda_impl_(std::make_unique<CUDA_impl>())
    {
        init(_point_cloud_sizes);
    }

    MetaPointCloud::MetaPointCloud(const MetaPointCloud& other)
        : cuda_impl_(std::make_unique<CUDA_impl>())
    {
        init(other.getPointcloudSizes());
        m_point_cloud_names = other.getCloudNames();

        for (uint16_t i = 0; i < other.getNumberOfPointclouds(); ++i)
            updatePointCloud(i, other.getPointCloud(i), other.getPointCloudSize(i), false);
        
        // copy all clouds on the device
        cuda_impl_->m_dev_ptr_to_accumulated_cloud = other.cuda_impl_->m_dev_ptr_to_accumulated_cloud;
    }

    // copy assignment
    MetaPointCloud& MetaPointCloud::operator=(const MetaPointCloud& other)
    {
        if (this != &other) // self-assignment check expected
        {
            destruct();
            init(other.getPointcloudSizes());
            m_point_cloud_names = other.getCloudNames();

            for (uint16_t i = 0; i < other.getNumberOfPointclouds(); ++i)
                updatePointCloud(i, other.getPointCloud(i), other.getPointCloudSize(i), false);

            // copy all clouds on the device
            cuda_impl_->m_dev_ptr_to_accumulated_cloud = other.cuda_impl_->m_dev_ptr_to_accumulated_cloud;
        }
        return *this;
    }


    bool MetaPointCloud::operator==(const MetaPointCloud& other) const
    {
        // Things are clear if self comparison:
        if (this == &other)
        {
            LOGGING_DEBUG_C(Gpu_voxels_helpers, MetaPointCloud, "Clouds are the same object." << icl_core::logging::endl);
            return true;
        }
        // Size and number of subclouds have to match:
        if (getAccumulatedPointcloudSize() != other.getAccumulatedPointcloudSize())
        {
            LOGGING_DEBUG_C(Gpu_voxels_helpers, MetaPointCloud, "Accumulated sizes do not match." << icl_core::logging::endl);
            return false;
        }
        if (getNumberOfPointclouds() != other.getNumberOfPointclouds())
        {
            LOGGING_DEBUG_C(Gpu_voxels_helpers, MetaPointCloud, "Number of sub-clouds do not match." << icl_core::logging::endl);
            return false;
        }

        // do the actual comparison:
        const bool ret = thrust::equal(cuda_impl_->m_dev_ptr_to_accumulated_cloud.begin(), cuda_impl_->m_dev_ptr_to_accumulated_cloud.end(), other.cuda_impl_->m_dev_ptr_to_accumulated_cloud.begin());
        CHECK_CUDA_ERROR();

        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

        if (!ret)
            LOGGING_DEBUG_C(Gpu_voxels_helpers, MetaPointCloud, "Clouds data is different!" << icl_core::logging::endl);

        return ret;
    }

    MetaPointCloud::MetaPointCloud(const std::vector<std::vector<Vector3f>>& point_clouds)
    {
        std::vector<uint32_t> point_cloud_sizes(point_clouds.size());
        for (size_t i = 0; i < point_clouds.size(); ++i)
            point_cloud_sizes[i] = static_cast<uint32_t>(point_clouds[i].size());

        init(point_cloud_sizes);
        for (uint16_t i = 0; i < static_cast<uint16_t>(point_clouds.size()); ++i)
            updatePointCloud(i, point_clouds[i], false);

        syncToDevice();
    }

    void MetaPointCloud::addCloud(uint32_t cloud_size)
    {
        auto new_sizes = getPointcloudSizes();
        new_sizes.push_back(cloud_size);

        // backup the current clouds
        auto tmp_clouds = m_accumulated_cloud;

        // Destruct current clouds on host and device
        destruct();
        // Allocate new mem
        init(new_sizes);

        // Restore previous data to new mem addresses
        m_accumulated_cloud = std::move(tmp_clouds);

        //delete[] tmp_clouds;
    }

    void MetaPointCloud::addCloud(const PointCloud& cloud, const std::string& name)
    {
        addCloud(cloud.getPointsDevice(), name);
    }

    void MetaPointCloud::addCloud(const std::vector<Vector3f>& cloud, bool sync, const std::string& name)
    {
        addCloud(cloud.data(), static_cast<uint32_t>(cloud.size()), sync, name);
    }

    //TODO:: this doesn't avoid names being inserted twice
    void MetaPointCloud::addCloud(const Vector3f* points, uint32_t pointcloud_size, bool sync, const std::string& name)
    {
        // Allocate mem and restore old data
        addCloud(pointcloud_size);

        // Copy the new cloud to host memory
        std::copy_n(points, m_point_clouds_local->cloud_sizes.back(), m_point_clouds_local->clouds_base_addresses.back());

        if (!name.empty())
            m_point_cloud_names[getNumberOfPointclouds() - 1] = name;

        if (sync)
            syncToDevice(getNumberOfPointclouds() - 1);
    }

    void MetaPointCloud::addCloud(const thrust::device_vector<Vector3f>& point_cloud, const std::string& name)
    {
        addCloud(static_cast<uint32_t>(point_cloud.size()));
        
        thrust::copy_n(point_cloud.begin(), m_point_clouds_local->cloud_sizes.back(), m_point_clouds_local->clouds_base_addresses.back());

        if (!name.empty())
            m_point_cloud_names[getNumberOfPointclouds() - 1] = name;

        // copy only the indicated cloud
        thrust::copy_n(point_cloud.begin(), m_point_clouds_local->cloud_sizes.back(), m_dev_ptrs_to_addrs.back());
    }

    std::string MetaPointCloud::getCloudName(uint16_t i) const
    {
	    const auto it = m_point_cloud_names.find(i);
        if (it != m_point_cloud_names.end())
        {
            return it->second;
        }
        LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud, "No name found for cloud index " << i << endl);
        return {};
    }

    std::optional<uint16_t> MetaPointCloud::getCloudNumber(const std::string& name) const
    {
        for (const auto& [cloudNumber, cloudName] : m_point_cloud_names)
        {
            if (name == cloudName)
                return cloudNumber;
        }
        LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud, "The name " << name << " is unknown" << endl);
        return std::nullopt;
    }

    bool MetaPointCloud::hasCloud(const std::string& name) const
    {
        for (const auto& cloudName : m_point_cloud_names | std::views::values)
        {
            if (name == cloudName)
                return true;
        }
        return false;
    }

    void MetaPointCloud::destruct()
    {
        thrust::device_free(cuda_impl_->m_dev_ptr_to_point_clouds_struct);
        cuda_impl_->m_dev_ptr_to_accumulated_cloud.clear();

        cuda_impl_->m_dev_ptr_to_cloud_sizes.clear();
        cuda_impl_->m_dev_ptr_to_clouds_base_addresses.clear();
        m_accumulated_cloud.clear();
        m_dev_point_clouds_local.reset();
        m_dev_ptrs_to_addrs.clear();
        m_point_clouds_local.reset();
    }

    MetaPointCloud::~MetaPointCloud()
    {
        destruct();
    }

    void MetaPointCloud::syncToDevice()
    {
        // copy all clouds to the device
        thrust::copy_n(m_point_clouds_local->clouds_base_addresses.front(), getAccumulatedPointcloudSize(), m_dev_ptrs_to_addrs.front());
    }

    void MetaPointCloud::syncToHost()
    {
        thrust::copy_n(m_dev_ptrs_to_addrs.front(), getAccumulatedPointcloudSize(), m_point_clouds_local->clouds_base_addresses.front());
    }

    void MetaPointCloud::syncToDevice(uint16_t cloud)
    {
        if (cloud < getNumberOfPointclouds())
        {
            // copy only the indicated cloud
            thrust::copy_n(m_point_clouds_local->clouds_base_addresses[cloud], m_point_clouds_local->cloud_sizes[cloud], m_dev_ptrs_to_addrs[cloud]);
        }
        else
        {
            LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                "Cloud " << cloud << "does not exist" << icl_core::logging::endl);
        }
    }

    void MetaPointCloud::updatePointCloud(uint16_t cloud, const std::vector<Vector3f>& pointcloud, bool sync)
    {
        updatePointCloud(cloud, pointcloud.data(), static_cast<uint32_t>(pointcloud.size()), sync);
    }

    void MetaPointCloud::updatePointCloud(uint16_t cloud, const PointCloud& pointcloud)
    {
        updatePointCloud(cloud, pointcloud.getPointsDevice());
    }

    void MetaPointCloud::updatePointCloud(uint16_t cloud, const thrust::device_vector<Vector3f>& pointcloud)
    {
        assert(getNumberOfPointclouds() >= cloud);
        
        if (pointcloud.size() == getPointCloudSize(cloud))
        {
            // Copy the cloud to host memory
            thrust::copy_n(pointcloud.begin(), pointcloud.size(), m_point_clouds_local->clouds_base_addresses[cloud]);
        }
        else
        {
            //    LOGGING_WARNING_C(Gpu_voxels_helpers, MetaPointCloud,
            //                    "Size of pointcloud changed! Rearanging memory" << icl_core::logging::endl);

            auto new_sizes = getPointcloudSizes();
            new_sizes[cloud] = static_cast<uint32_t>(pointcloud.size());

            // backup the current clouds
            std::vector<std::vector<Vector3f>> tmp_clouds;
            for (uint16_t i = 0; i < getNumberOfPointclouds(); i++)
            {
                if (i != cloud)
                {
                    const auto original_cloud = getPointCloud(i);
                    std::vector<Vector3f> tmp_cloud(original_cloud.begin(), original_cloud.end());

                    tmp_clouds.emplace_back(std::move(tmp_cloud));
                }
                else {
                    // skip the modified cloud
                }
            }
            destruct();      // Destruct current clouds on host and device
            init(new_sizes); // Allocate new mem
            // Restore previous data to new mem addresses
            uint16_t j = 0;
            for (uint16_t i = 0; i < getNumberOfPointclouds(); i++)
            {
                if (i != cloud)
                {
                    std::copy_n(tmp_clouds[j].begin(), getPointCloudSize(i), m_point_clouds_local->clouds_base_addresses[i]);
                    ++j;
                }
                else 
                {
                    thrust::copy_n(pointcloud.begin(), getPointCloudSize(i), m_point_clouds_local->clouds_base_addresses[i]);
                }
            }
        }
        thrust::copy_n(pointcloud.begin(), m_point_clouds_local->cloud_sizes[cloud], m_dev_ptrs_to_addrs[cloud]);
    }

    void MetaPointCloud::updatePointCloud(const std::string& cloud_name, const std::vector<Vector3f>& pointcloud, bool sync)
    {
        const auto cloud_id = getCloudNumber(cloud_name);
        if (cloud_id.has_value())
            updatePointCloud(cloud_id.value(), pointcloud.data(), static_cast<uint32_t>(pointcloud.size()), sync);
    }

    void MetaPointCloud::updatePointCloud(uint16_t cloud, const Vector3f* pointcloud, uint32_t pointcloud_size, bool sync)
    {
        assert(getNumberOfPointclouds() >= cloud);

        if (pointcloud_size == m_point_clouds_local->cloud_sizes[cloud])
        {
            // Copy the cloud to host memory
            std::copy_n(pointcloud, pointcloud_size, m_point_clouds_local->clouds_base_addresses[cloud]);
        }
        else
        {
            //    LOGGING_WARNING_C(Gpu_voxels_helpers, MetaPointCloud,
            //                    "Size of pointcloud changed! Rearanging memory" << icl_core::logging::endl);

            auto new_sizes = getPointcloudSizes();
            new_sizes[cloud] = pointcloud_size;

            // backup the current clouds
            std::vector<std::vector<Vector3f>> tmp_clouds;
            for (uint16_t i = 0; i < getNumberOfPointclouds(); i++)
            {
                if (i != cloud)
                {
                    const auto original_cloud = getPointCloud(i);
                    std::vector<Vector3f> tmp_cloud(original_cloud.begin(), original_cloud.end());

                    tmp_clouds.emplace_back(std::move(tmp_cloud));
                }
                else {
                    // skip the modified cloud
                }
            }
            destruct();      // Destruct current clouds on host and device
            init(new_sizes); // Allocate new mem
            // Restore previous data to new mem addresses
            uint16_t j = 0;
            for (uint16_t i = 0; i < getNumberOfPointclouds(); i++)
            {
                if (i != cloud)
                {
                    std::copy_n(tmp_clouds[j].begin(), getPointCloudSize(i), m_point_clouds_local->clouds_base_addresses[i]);
                    ++j;
                }
                else 
                {
                    std::copy_n(pointcloud, getPointCloudSize(i), m_point_clouds_local->clouds_base_addresses[i]);
                }
            }
        }
        if (sync)
        {
            syncToDevice(cloud);
        }
    }

    void MetaPointCloud::updatePointCloud(uint16_t cloud, const std::span<Vector3f>& pointcloud, uint32_t pointcloud_size, bool sync)
    {
        updatePointCloud(cloud, pointcloud.data(), pointcloud_size, sync);
    }

    uint16_t MetaPointCloud::getNumberOfPointclouds() const
    {
        return m_point_clouds_local->num_clouds;
    }

    uint32_t MetaPointCloud::getPointCloudSize(uint16_t cloud) const
    {
        return m_point_clouds_local->cloud_sizes[cloud];
    }

    uint32_t MetaPointCloud::getAccumulatedPointcloudSize() const
    {
        return m_accumulated_cloud.size();
    }

    bool MetaPointCloud::empty() const
    {
        return m_accumulated_cloud.empty();
    }

    const thrust::host_vector<uint32_t>& MetaPointCloud::getPointcloudSizes() const
    {
        return m_point_clouds_local->cloud_sizes;
    }

    std::span<Vector3f> MetaPointCloud::getPointCloud(uint16_t cloud) const
    {
        assert(getNumberOfPointclouds() >= cloud);

        return { m_point_clouds_local->clouds_base_addresses[cloud], m_point_clouds_local->cloud_sizes[cloud] };
    }

    thrust::device_ptr<MetaPointCloudStruct> MetaPointCloud::getDevicePointer() const
    {
        return cuda_impl_->m_dev_ptr_to_point_clouds_struct;
    }

    const std::map<uint16_t, std::string>& MetaPointCloud::getCloudNames() const
    {
        return m_point_cloud_names;
    }

    thrust::device_ptr<const MetaPointCloudStruct> MetaPointCloud::getDeviceConstPointer() const
    {
        return cuda_impl_->m_dev_ptr_to_point_clouds_struct;
    }

    void MetaPointCloud::debugPointCloud() const
    {
        std::cout << "================== hostMetaPointCloud DBG ================== \n";

        std::cout << "hostDebugMetaPointCloud DBG: NumClouds: " << getNumberOfPointclouds() << std::endl;

        std::cout << "hostDebugMetaPointCloud DBG: m_dev_ptr_to_clouds_base_addresses: " << static_cast<void*>(m_point_clouds_local->clouds_base_addresses.data()) << std::endl;

        for (uint16_t i = 0; i < m_point_clouds_local->num_clouds; i++)
        {
            printf("hostDebugMetaPointCloud DBG: '%s' CloudSize[%d]: %d, clouds_base_addresses[%d]: %p \n",
                this->getCloudName(i).c_str(),
                i, m_point_clouds_local->cloud_sizes[i],
                i, &(*m_point_clouds_local->clouds_base_addresses[i]));

            if (getPointCloudSize(i) == 0)
                continue;

            auto pcl = getPointCloud(i);
            auto [min_xyz, max_xyz] = std::ranges::minmax_element(pcl, [](const Vector3f& v0, const Vector3f& v1)
            {
                    return v0.x() < v1.x() || v0.y() < v1.x() || v0.z() < v1.z();
            });

            printf("hostDebugMetaPointCloud DBG: Cloud %d bounds: Min[%f, %f, %f], Max[%f, %f, %f] \n",
                i, min_xyz->x(), min_xyz->y(), min_xyz->z(), max_xyz->x(), max_xyz->y(), max_xyz->z());
        }

        printf("================== END hostDebugMetaPointCloud DBG ================== \n");

        kernelDebugMetaPointCloud<<<1, 1>>>(cuda_impl_->m_dev_ptr_to_point_clouds_struct.get());
        CHECK_CUDA_ERROR();
    }

    void MetaPointCloud::transformSelfSubCloud(uint8_t subcloud_to_transform, const Matrix4f& transformation) const
    {
        transformSubCloud(subcloud_to_transform, transformation, *this);
    }

    void MetaPointCloud::transformSelf(const Matrix4f& transformation)
    {
        transform(transformation, *this);
    }

    void MetaPointCloud::transform(const Matrix4f& transformation, MetaPointCloud& transformed_cloud) const
    {
        if (getAccumulatedPointcloudSize() != transformed_cloud.getAccumulatedPointcloudSize() ||
            getNumberOfPointclouds() != transformed_cloud.getNumberOfPointclouds())
        {
            LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                "Size of target pointcloud does not match local pointcloud. Not transforming!" << icl_core::logging::endl);
            return;
        }
        if (empty())
            return;
        
        thrust::transform(thrust::cuda::par_nosync, *m_dev_ptrs_to_addrs.begin(), *m_dev_ptrs_to_addrs.end(), 
            *transformed_cloud.m_dev_ptrs_to_addrs.begin(), KernelTransform(transformation));
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void MetaPointCloud::transformSubCloud(uint16_t subcloud_to_transform, const Matrix4f& transformation, const MetaPointCloud& transformed_cloud) const
    {
        if (empty())
            return;

        if (m_point_clouds_local->cloud_sizes[subcloud_to_transform] != transformed_cloud.m_point_clouds_local->cloud_sizes[subcloud_to_transform])
        {
            LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                "Size of target sub-pointcloud does not match local pointcloud. Not transforming!" << icl_core::logging::endl);
            return;
        }

        const auto begin = m_dev_ptrs_to_addrs[subcloud_to_transform];
        const auto end = begin + m_point_clouds_local->cloud_sizes[subcloud_to_transform];
        //const auto device_cloud = getPointCloudDevice(subcloud_to_transform);
        

        //auto t0 = std::chrono::steady_clock::now();
        //for (int i = 0; i < 30000; ++i)
        //{
        thrust::transform(thrust::cuda::par_nosync, begin, end,
            transformed_cloud.m_dev_ptrs_to_addrs[subcloud_to_transform], KernelTransform(transformation));
        //}
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        /*auto tt0 = std::chrono::steady_clock::now() - t0;

        auto m_transformation_dev = thrust::device_malloc<Matrix4f>(1);
        HANDLE_CUDA_ERROR(cudaMemcpy(m_transformation_dev.get(), &transformation, sizeof(Matrix4f), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();

        
        uint32_t m_blocks, m_threads_per_block;
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < 30000; ++i)
        {
            
            computeLinearLoad(getPointCloudSize(subcloud_to_transform), m_blocks, m_threads_per_block);
            //cudaDeviceSynchronize();
            // transform the cloud via Kernel.
            kernelTransformCloud<<<m_blocks, m_threads_per_block>>>
                (m_transformation_dev.get(),
                    m_dev_ptrs_to_addrs[subcloud_to_transform].get(),
                    transformed_cloud.m_dev_ptrs_to_addrs[subcloud_to_transform].get(),
                    m_point_clouds_local->cloud_sizes[subcloud_to_transform]);
            CHECK_CUDA_ERROR();
        }
        cudaDeviceSynchronize();
        auto tt1 = std::chrono::steady_clock::now() - t1;

        std::cout << "[0]: " << std::chrono::duration<double>(tt0) << std::endl;
        std::cout << "[1]: " << std::chrono::duration<double>(tt1) << std::endl;
    	std::cout << std::chrono::duration<double>(tt0).count() / std::chrono::duration<double>(tt1).count() << std::endl << std::endl;
        //HANDLE_CUDA_ERROR(cudaDeviceSynchronize());*/
    }
    void MetaPointCloud::transformSubClouds(uint16_t startcloud_to_transform, const std::vector<Matrix4f>& transformations, const MetaPointCloud& transformed_cloud) const
    {
        if (empty())
            return;

        for (uint16_t i = startcloud_to_transform; i < transformations.size(); ++i)
        {
            if (m_point_clouds_local->cloud_sizes[i] != transformed_cloud.m_point_clouds_local->cloud_sizes[i])
            {
                LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                    "Size of target sub-pointcloud does not match local pointcloud. Not transforming!" << icl_core::logging::endl);
                return;
            }
        }
        
        std::vector<cudaStream_t> streams(transformations.size());
        for (auto& stream : streams)
            cudaStreamCreate(&stream);

        for (size_t i = 0; i < transformations.size(); ++i)
        {
            const uint16_t current_cloud = i + startcloud_to_transform;

            const auto begin = m_dev_ptrs_to_addrs[current_cloud];
            const auto end = begin + m_point_clouds_local->cloud_sizes[current_cloud];

            thrust::transform(thrust::cuda::par_nosync.on(streams[i]), begin, end,
                transformed_cloud.m_dev_ptrs_to_addrs[current_cloud], KernelTransform(transformations[i]));
        }
        
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        for (const auto& stream : streams)
            cudaStreamDestroy(stream);
    }

    void MetaPointCloud::transformSubClouds(const std::list<std::tuple<uint16_t, Matrix4f>>& subcloud_transforms, const MetaPointCloud& transformed_cloud) const
    {
        if (empty())
            return;

        for (const auto& subcloud_id : subcloud_transforms | std::views::elements<0>)
        {
            if (m_point_clouds_local->cloud_sizes[subcloud_id] != transformed_cloud.m_point_clouds_local->cloud_sizes[subcloud_id])
            {
                LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                    "Size of target sub-pointcloud does not match local pointcloud. Not transforming!" << icl_core::logging::endl);
                return;
            }
        }

        std::vector<cudaStream_t> streams(subcloud_transforms.size());
        for (auto& stream : streams)
            cudaStreamCreate(&stream);

        size_t i = 0;
        for (const auto& [subcloud_id, transform] : subcloud_transforms)
        {
            const auto begin = m_dev_ptrs_to_addrs[subcloud_id];
            const auto end = begin + m_point_clouds_local->cloud_sizes[subcloud_id];

            thrust::transform(thrust::cuda::par_nosync.on(streams[i++]), begin, end,
                transformed_cloud.m_dev_ptrs_to_addrs[subcloud_id], KernelTransform(transform));
        }

        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        for (const auto& stream : streams)
            cudaStreamDestroy(stream);
    }
} // end of ns gpu_voxels