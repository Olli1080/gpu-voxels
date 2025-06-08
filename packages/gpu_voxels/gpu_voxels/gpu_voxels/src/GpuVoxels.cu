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
 * \date    2014-06-08
 *
 */
//----------------------------------------------------------------------
#include "GpuVoxels.h"

#include <utility>

#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <gpu_voxels/helpers/GeometryGeneration.h>

#ifdef GPUVOXEL_VISUALIZE
#include <gpu_voxels/vis_interface/VisVoxelMap.h>
#include <gpu_voxels/vis_interface/VisTemplateVoxelList.h>
#include <gpu_voxels/vis_interface/VisPrimitiveArray.h>
#include <gpu_voxels/octree/VisNTree.h>
#endif

#include <gpu_voxels/core/ManagedPrimitiveArray.h>
#include <gpu_voxels/core/ManagedMap.h>

namespace gpu_voxels {

    GpuVoxels::GpuVoxels()
        :
        m_dim(Vector3ui::Zero()),
        m_voxel_side_length(0)
    {
        // Check for valid GPU:
        if (!cuTestAndInitDevice())
        {
            exit(123);
        }
    }

    GpuVoxels::~GpuVoxels()
    {
        // as the objects are shared pointers, they get deleted by this.
        m_managed_maps.clear();
        m_managed_robots.clear();
        m_managed_primitive_arrays.clear();
    }

    void GpuVoxels::initialize(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z, const float voxel_side_length)
    {
        if (m_dim.x() == 0 || m_dim.y() == 0 || m_dim.z() == 0 || m_voxel_side_length == 0)
        {
            m_dim.x() = dim_x;
            m_dim.y() = dim_y;
            m_dim.z() = dim_z;
            m_voxel_side_length = voxel_side_length;
        }
        else
        {
            LOGGING_WARNING(Gpu_voxels, "Do not try to initialize GpuVoxels multiple times. Parameters remain unchanged." << endl);
        }
    }

    std::weak_ptr<GpuVoxels> GpuVoxels::masterPtr = std::weak_ptr<GpuVoxels>();

    GpuVoxelsSharedPtr GpuVoxels::getInstance()
    {
        std::shared_ptr<GpuVoxels> temp = gpu_voxels::GpuVoxels::masterPtr.lock();
        if (!temp)
        {
            temp.reset(new GpuVoxels());
            gpu_voxels::GpuVoxels::masterPtr = temp;
        }
        return temp;
    }

    bool GpuVoxels::addPrimitives(const primitive_array::PrimitiveType prim_type, const std::string& array_name)
    {
        // check if array with same name already exists
        const auto it = m_managed_primitive_arrays.find(array_name);
        if (it != m_managed_primitive_arrays.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' already exists." << endl);
            return false;
        }

        auto orig_prim_array = std::make_unique<primitive_array::PrimitiveArray>(m_dim, m_voxel_side_length, prim_type);
#ifdef GPUVOXEL_VISUALIZE
        auto vis_prim_array = std::make_unique<VisPrimitiveArray>(orig_prim_array.get(), array_name);
#endif
        const auto primitive_array_shared_ptr = primitive_array::PrimitiveArraySharedPtr(orig_prim_array.release());


#ifdef GPUVOXEL_VISUALIZE
        const auto vis_primitives_shared_ptr = VisProviderSharedPtr(vis_prim_array.release());
#else
        const auto vis_primitives_shared_ptr = std::make_shared<VisProvider>();
#endif
        
        m_managed_primitive_arrays.emplace(array_name,
            ManagedPrimitiveArray(primitive_array_shared_ptr, vis_primitives_shared_ptr));
        return true;
    }

    bool GpuVoxels::delPrimitives(const std::string& array_name)
    {
        const auto it = m_managed_primitive_arrays.find(array_name);
        if (it == m_managed_primitive_arrays.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
            return false;
        }
        m_managed_primitive_arrays.erase(it);
        return true;
    }

    bool GpuVoxels::modifyPrimitives(const std::string& array_name, const std::vector<Vector4f>& prim_positions)
    {
        const auto it = m_managed_primitive_arrays.find(array_name);
        if (it == m_managed_primitive_arrays.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
            return false;
        }
        it->second.prim_array_shared_ptr->setPoints(prim_positions);
        return true;
    }

    bool GpuVoxels::modifyPrimitives(const std::string& array_name, const std::vector<Vector4i>& prim_positions)
    {
        const auto it = m_managed_primitive_arrays.find(array_name);
        if (it == m_managed_primitive_arrays.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
            return false;
        }
        it->second.prim_array_shared_ptr->setPoints(prim_positions);
        return true;
    }

    bool GpuVoxels::modifyPrimitives(const std::string& array_name, const std::vector<Vector3f>& prim_positions, const float& diameter)
    {
        const auto it = m_managed_primitive_arrays.find(array_name);
        if (it == m_managed_primitive_arrays.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
            return false;
        }
        it->second.prim_array_shared_ptr->setPoints(prim_positions, diameter);
        return true;
    }

    bool GpuVoxels::modifyPrimitives(const std::string& array_name, const std::vector<Vector3i>& prim_positions, const uint32_t& diameter)
    {
        const auto it = m_managed_primitive_arrays.find(array_name);
        if (it == m_managed_primitive_arrays.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives array with name '" << array_name << "' not found." << endl);
            return false;
        }
        it->second.prim_array_shared_ptr->setPoints(prim_positions, diameter);
        return true;
    }

    GpuVoxelsMapSharedPtr GpuVoxels::addMap(const MapType map_type, const std::string& map_name)
    {
        GpuVoxelsMapSharedPtr map_shared_ptr;
        VisProviderSharedPtr vis_map_shared_ptr;

        // check if map with same name already exists
        const auto it = m_managed_maps.find(map_name);
        if (it != m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' already exists." << endl);

            return map_shared_ptr;  // null-initialized shared_ptr!
        }

        switch (map_type)
        {
        case MT_PROBAB_VOXELMAP:
        {
            auto orig_map = std::make_unique<voxelmap::ProbVoxelMap>(m_dim, m_voxel_side_length, MT_PROBAB_VOXELMAP);

#ifdef GPUVOXEL_VISUALIZE
            auto vis_map = std::make_unique<VisVoxelMap>(orig_map.get(), map_name);
#endif

            map_shared_ptr = GpuVoxelsMapSharedPtr(orig_map.release());

#ifdef GPUVOXEL_VISUALIZE
            vis_map_shared_ptr = VisProviderSharedPtr(vis_map.release());
#endif
            break;
        }

        case MT_BITVECTOR_VOXELLIST:
        {
            auto orig_list = std::make_unique<voxellist::BitVectorVoxelList>(m_dim, m_voxel_side_length, MT_BITVECTOR_VOXELLIST);

#ifdef GPUVOXEL_VISUALIZE
            auto vis_list = std::make_unique<VisTemplateVoxelList<BitVectorVoxel, uint32_t>>(orig_list.get(), map_name);
#endif

            map_shared_ptr = GpuVoxelsMapSharedPtr(orig_list.release());

#ifdef GPUVOXEL_VISUALIZE
            vis_map_shared_ptr = VisProviderSharedPtr(vis_list.release());
#endif
            break;
        }

        case MT_BITVECTOR_OCTREE:
        {
            auto ntree = std::make_unique<NTree::GvlNTreeDet>(m_voxel_side_length, MT_BITVECTOR_OCTREE);

#ifdef GPUVOXEL_VISUALIZE
            auto vis_map = std::make_unique<NTree::VisNTreeDet>(ntree.get(), map_name);
#endif

            map_shared_ptr = GpuVoxelsMapSharedPtr(ntree.release());

#ifdef GPUVOXEL_VISUALIZE
            vis_map_shared_ptr = VisProviderSharedPtr(vis_map.release());
#endif
            break;
        }

        case MT_BITVECTOR_MORTON_VOXELLIST:
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED << endl);
            throw std::exception(GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED.c_str());
        }

        case MT_BITVECTOR_VOXELMAP:
        {
            auto orig_map = std::make_unique<voxelmap::BitVectorVoxelMap>(m_dim, m_voxel_side_length, MT_BITVECTOR_VOXELMAP);
#ifdef GPUVOXEL_VISUALIZE
            auto vis_map = std::make_unique<VisVoxelMap>(orig_map.get(), map_name);
#endif

            map_shared_ptr = GpuVoxelsMapSharedPtr(orig_map.release());

#ifdef GPUVOXEL_VISUALIZE
            vis_map_shared_ptr = VisProviderSharedPtr(vis_map.release());
#endif
            break;
        }

        case MT_COUNTING_VOXELLIST:
        {
            auto orig_list = std::make_unique<voxellist::CountingVoxelList>(m_dim, m_voxel_side_length, MT_COUNTING_VOXELLIST);

#ifdef GPUVOXEL_VISUALIZE
            auto vis_list = std::make_unique<VisTemplateVoxelList<CountingVoxel, uint32_t>>(orig_list.get(), map_name);
#endif

            map_shared_ptr = GpuVoxelsMapSharedPtr(orig_list.release());

#ifdef GPUVOXEL_VISUALIZE
            vis_map_shared_ptr = VisProviderSharedPtr(vis_list.release());
#endif
            break;
        }

        case MT_PROBAB_VOXELLIST:
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED << endl);
            throw std::exception(GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED.c_str());
        }

        case MT_PROBAB_OCTREE:
        {
            auto ntree = std::make_unique<NTree::GvlNTreeProb>(m_voxel_side_length, MT_PROBAB_OCTREE);
#ifdef GPUVOXEL_VISUALIZE
            auto vis_map = std::make_unique<NTree::VisNTreeProb>(ntree.get(), map_name);
#endif
            map_shared_ptr = GpuVoxelsMapSharedPtr(ntree.release());
#ifdef GPUVOXEL_VISUALIZE
            vis_map_shared_ptr = VisProviderSharedPtr(vis_map.release());
#endif
            break;
        }

        case MT_PROBAB_MORTON_VOXELLIST:
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED << endl);
            throw std::exception(GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED.c_str());
        }

        case MT_DISTANCE_VOXELMAP:
        {
            auto orig_map = std::make_unique<voxelmap::DistanceVoxelMap>(m_dim, m_voxel_side_length, MT_DISTANCE_VOXELMAP);
#ifdef GPUVOXEL_VISUALIZE
            auto vis_map = std::make_unique<VisVoxelMap>(orig_map.get(), map_name);
#endif
            map_shared_ptr = GpuVoxelsMapSharedPtr(orig_map.release());

#ifdef GPUVOXEL_VISUALIZE
            vis_map_shared_ptr = VisProviderSharedPtr(vis_map.release());
#endif
            break;
        }

        default:
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "THIS TYPE OF MAP IS UNKNOWN!" << endl);
            throw std::exception(GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED.c_str());
        }
        }

        if (map_shared_ptr)
        {
            m_managed_maps.emplace(map_name, ManagedMap(map_shared_ptr, vis_map_shared_ptr));

            // sanity checking, that nothing went wrong:
            CHECK_CUDA_ERROR();
            return map_shared_ptr;
        }
        else
        {
            throw std::exception("Map was not set");
        }


    }

    bool GpuVoxels::delMap(const std::string& map_name)
    {
        const auto it = m_managed_maps.find(map_name);
        if (it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
            return false;
        }
        m_managed_maps.erase(it);
        return true;
    }

    GpuVoxelsMapSharedPtr GpuVoxels::getMap(const std::string& map_name)
    {
        const auto it = m_managed_maps.find(map_name);
        if (it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
            return {};
        }
        return m_managed_maps.find(map_name)->second.map_shared_ptr;
    }

    // ---------- Robot Stuff ------------
    RobotInterfaceSharedPtr GpuVoxels::getRobot(const std::string& rob_name)
    {
        const auto it = m_managed_robots.find(rob_name);
        if (it == m_managed_robots.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Robot with name '" << rob_name << "' not found." << endl);
            return {};
        }
        return it->second;
    }
    

#ifdef _BUILD_GVL_WITH_URDF_SUPPORT_
    bool GpuVoxels::addRobot(const std::string& robot_name, const std::string& path_to_urdf_file, const bool use_model_path)
    {
        // check if robot with same name already exists
        ManagedRobotsIterator it = m_managed_robots.find(robot_name);
        if (it != m_managed_robots.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Robot with name '" << robot_name << "' already exists." << endl);
            return false;
        }

        m_managed_robots.emplace(
                robot_name, RobotInterfaceSharedPtr(new robot::UrdfRobot(m_voxel_side_length, path_to_urdf_file, use_model_path)));

        return true;
    }
#endif

    bool GpuVoxels::updateRobotPart(const std::string& robot_name, const std::string& link_name, 
        const std::vector<Vector3f>& pointcloud)
    {
        const auto rob = getRobot(robot_name);
        if (!rob)
            return false;

        rob->updatePointcloud(link_name, pointcloud);
        return true;
    }

    bool GpuVoxels::setRobotConfiguration(const std::string& robot_name,
                                          const robot::JointValueMap& jointmap)
    {
        const auto rob = getRobot(robot_name);
        if (!rob)
            return false;

        rob->setConfiguration(jointmap);
        return true;
    }

    bool GpuVoxels::setRobotBaseTransformation(const std::string& robot_name, const Matrix4f& transformation)
    {
        const auto rob = getRobot(robot_name);
        if (!rob)
            return false;

        rob->setBaseTransformation(transformation);
        return true;
    }

    bool GpuVoxels::getRobotTransformation(const std::string& robot_name, size_t idx, Matrix4f& transformation)
    {
        const auto rob = getRobot(robot_name);
        if (!rob)
            return false;

        transformation = rob->getTransform(idx);
        return true;
    }

    bool GpuVoxels::getRobotConfiguration(const std::string& robot_name, robot::JointValueMap& jointmap)
    {
        const auto rob = getRobot(robot_name);
        if (!rob)
            return false;

        rob->getConfiguration(jointmap);
        return true;
    }

    bool GpuVoxels::insertPointCloudFromFile(const std::string& map_name, const std::string& path,
        const std::filesystem::path& model_path, const BitVoxelMeaning voxel_meaning,
        const bool shift_to_zero, const Vector3f& offset_XYZ, const float scaling)
    {
        const auto map_it = m_managed_maps.find(map_name);
        if (map_it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
            return false;
        }

        return map_it->second.map_shared_ptr->insertPointCloudFromFile(path, model_path, voxel_meaning,
            shift_to_zero, offset_XYZ, scaling);
    }

    bool GpuVoxels::insertPointCloudIntoMap(const PointCloud& cloud, const std::string& map_name, const BitVoxelMeaning voxel_meaning)
    {
        const auto map_it = m_managed_maps.find(map_name);
        if (map_it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
            return false;
        }

        map_it->second.map_shared_ptr->insertPointCloud(cloud, voxel_meaning);

        return true;
    }

    bool GpuVoxels::insertPointCloudIntoMap(const std::vector<Vector3f>& cloud, const std::string& map_name, const BitVoxelMeaning voxel_meaning)
    {
        const auto map_it = m_managed_maps.find(map_name);
        if (map_it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
            return false;
        }

        map_it->second.map_shared_ptr->insertPointCloud(cloud, voxel_meaning);

        return true;
    }

    bool GpuVoxels::insertMetaPointCloudIntoMap(const MetaPointCloud& cloud, const std::string& map_name, const std::vector<BitVoxelMeaning>& voxel_meanings)
    {
        const auto map_it = m_managed_maps.find(map_name);
        if (map_it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
            return false;
        }

        map_it->second.map_shared_ptr->insertMetaPointCloud(cloud, voxel_meanings);

        return true;
    }

    bool GpuVoxels::insertMetaPointCloudIntoMap(const MetaPointCloud& cloud, const std::string& map_name, const BitVoxelMeaning voxel_meaning)
    {
        const auto map_it = m_managed_maps.find(map_name);
        if (map_it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
            return false;
        }

        map_it->second.map_shared_ptr->insertMetaPointCloud(cloud, voxel_meaning);

        return true;
    }

    bool GpuVoxels::insertRobotIntoMap(const std::string& robot_name, const std::string& map_name, const BitVoxelMeaning voxel_meaning)
    {
        const auto rob = getRobot(robot_name);
        if (!rob)
            return false;

        const auto map_it = m_managed_maps.find(map_name);
        if (map_it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
            return false;
        }

        map_it->second.map_shared_ptr->insertMetaPointCloud(*rob->getTransformedClouds(), voxel_meaning);

        return true;
    }

    bool GpuVoxels::insertRobotIntoMapSelfCollAware(const std::string& robot_name, const std::string& map_name,
                                                    const std::vector<BitVoxelMeaning>& voxel_meanings,
                                                    const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks,
                                                    BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
    {
        const auto rob = getRobot(robot_name);
        if (!rob)
            return false;

        const auto map_it = m_managed_maps.find(map_name);
        if (map_it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
            return false;
        }

        return map_it->second.map_shared_ptr->insertMetaPointCloudWithSelfCollisionCheck(rob->getTransformedClouds(),
            voxel_meanings, collision_masks, colliding_meanings);

    }

    bool GpuVoxels::insertBoxIntoMap(const Vector3f& corner_min, const Vector3f& corner_max, const std::string& map_name, const BitVoxelMeaning voxel_meaning, uint16_t points_per_voxel)
    {
        const auto map_it = m_managed_maps.find(map_name);
        if (map_it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Could not find map '" << map_name << "'" << endl);
            return false;
        }

        const float delta = m_voxel_side_length / static_cast<float>(points_per_voxel);

        const std::vector<Vector3ui> coordinates = geometry_generation::createBoxOfPoints(corner_min, corner_max, delta, m_voxel_side_length);
        map_it->second.map_shared_ptr->insertCoordinateList(coordinates, voxel_meaning);

        return true;
    }

    bool GpuVoxels::clearMap(const std::string& map_name)
    {
        const auto it = m_managed_maps.find(map_name);
        if (it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
            return false;
        }
        it->second.map_shared_ptr->clearMap();
        return true;
    }

    bool GpuVoxels::clearMap(const std::string& map_name, BitVoxelMeaning voxel_meaning)
    {
        const auto it = m_managed_maps.find(map_name);
        if (it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
            return false;
        }
        it->second.map_shared_ptr->clearBitVoxelMeaning(voxel_meaning);
        return true;
    }

    bool GpuVoxels::visualizeMap(const std::string& map_name, const bool force_repaint)
    {
        const auto it = m_managed_maps.find(map_name);
        if (it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
            return false;
        }
#ifdef GPUVOXEL_VISUALIZE
        return it->second.vis_provider_shared_ptr.get()->visualize(force_repaint);
#else
        return false;
#endif
    }

    bool GpuVoxels::visualizePrimitivesArray(const std::string& prim_array_name, const bool force_repaint)
    {
        const auto it = m_managed_primitive_arrays.find(prim_array_name);
        if (it == m_managed_primitive_arrays.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Primitives Array with name '" << prim_array_name << "' not found." << endl);
            return false;
        }
#ifdef GPUVOXEL_VISUALIZE
        return it->second.vis_provider_shared_ptr.get()->visualize(force_repaint);
#else
        return false;
#endif
    }

    VisProvider* GpuVoxels::getVisualization(const std::string& map_name)
    {
        const auto it = m_managed_maps.find(map_name);
        if (it == m_managed_maps.end())
        {
            LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
            return nullptr;
        }
        return it->second.vis_provider_shared_ptr.get();
    }

    void GpuVoxels::getDimensions(uint32_t& dim_x, uint32_t& dim_y, uint32_t& dim_z) const
    {
        dim_x = m_dim.x();
        dim_y = m_dim.y();
        dim_z = m_dim.z();
    }

    void GpuVoxels::getDimensions(Vector3ui& dim) const
    {
        dim = m_dim;
    }

    void GpuVoxels::getVoxelSideLength(float& voxel_side_length) const
    {
        voxel_side_length = m_voxel_side_length;
    }

}