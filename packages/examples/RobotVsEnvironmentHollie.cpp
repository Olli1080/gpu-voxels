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
 *
 * This example program shows a simple collision check
 * between an animated Robot and a Static Map.
 * Be sure to press g in the viewer to draw the whole map.
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <csignal>

#include <icl_core_config/Config.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

#include <mutex>
#include <ranges>
#include <set>

#include <pcl/point_cloud.h>
#include <pcl/impl/point_types.hpp>

#include <pcl/io/ply_io.h>

using namespace gpu_voxels;

GpuVoxelsSharedPtr gvl;

std::atomic<bool> running = true;

void ctrlchandler(int)
{
	running = false;
}
void killhandler(int)
{
	running = false;
}

struct Vec3
{
	uint32_t x, y, z;

    friend bool operator<(const Vec3& l, const Vec3& r)
    {
        return std::tie(l.x, l.y, l.z)
            < std::tie(r.x, r.y, r.z); // keep the same order
    }
};

int main(int argc, char* argv[])
{
    signal(SIGINT, ctrlchandler);
    signal(SIGTERM, killhandler);

    icl_core::logging::initialize(argc, argv);

    const char* robot_id = "myRobotMap";
    const char* env_id = "myEnvironmentMap";

    /*
     * First, we generate an API class, which defines the
     * volume of our space and the resolution.
     * Be careful here! The size is limited by the memory
     * of your GPU. Even if an empty Octree is small, a
     * Voxelmap will always require the full memory.
     */
    gvl = GpuVoxels::getInstance();
    gvl->initialize(200, 200, 100, 0.015f);

    /*
     * Now we add a map, that will represent the robot.
     * The robot is inserted with deterministic poses,
     * so a deterministic map is sufficient here.
     */
    gvl->addMap(MT_BITVECTOR_VOXELMAP, robot_id);


    /*
     * A second map will represent the environment.
     * As it is captured by a sensor, this map is probabilistic.
     * We also have an a priori static map file in a PCD, so we
     * also load that into the map.
     * The PCD file is (in this example) retrieved via an environment variable
     * to access the directory where model files are stored in:
     * GPU_VOXELS_MODEL_PATH
     * The additional (optional) params shift the map to zero and then add
     * a offset to the loaded pointcloud.
     */
    gvl->addMap(MT_BITVECTOR_OCTREE, env_id);

    if (!gvl->insertPointCloudFromFile(env_id, "pointcloud_0002.pcd", true,
        eBVM_OCCUPIED, true, Vector3f(-6.f, -7.3f, 0.f)))
    {
        LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
    }

    /*
     * Of course, we need a robot. At this point, you can choose between
     * describing your robot via ROS URDF or via conventional DH parameter.
     * In this example, we simply hardcode a DH robot:
     */

     // First, we load the robot geometry which contains 9 links with 7 geometries:
     // Geometries are required to have the same names as links, if they should get transformed.
    std::vector<std::string> linknames(10);
    std::vector<std::string> paths_to_pointclouds(7);
    linknames[0] = "z_translation";
    linknames[1] = "y_translation";
    linknames[2] = "x_translation";
    linknames[3] = paths_to_pointclouds[0] = "hollie/arm_0_link.xyz";
    linknames[4] = paths_to_pointclouds[1] = "hollie/arm_1_link.xyz";
    linknames[5] = paths_to_pointclouds[2] = "hollie/arm_2_link.xyz";
    linknames[6] = paths_to_pointclouds[3] = "hollie/arm_3_link.xyz";
    linknames[7] = paths_to_pointclouds[4] = "hollie/arm_4_link.xyz";
    linknames[8] = paths_to_pointclouds[5] = "hollie/arm_5_link.xyz";
    linknames[9] = paths_to_pointclouds[6] = "hollie/arm_6_link.xyz";

    std::vector<robot::DHParameters> dh_params(10);
    // _d,  _theta,  _a,   _alpha, _value, _type
    dh_params[0] = robot::DHParameters(0.0, 0.0, 0.0, -1.5708f, 0.0, robot::PRISMATIC); // Params for Y translation
    dh_params[1] = robot::DHParameters(0.0, -1.5708f, 0.0, -1.5708f, 0.0, robot::PRISMATIC); // Params for X translation
    dh_params[2] = robot::DHParameters(0.0, 1.5708f, 0.0, 1.5708f, 0.0, robot::PRISMATIC); // Params for first Robot axis (visualized by 0_link)
    dh_params[3] = robot::DHParameters(0.0, 1.5708f, 0.0, 1.5708f, 0.0, robot::REVOLUTE);  // Params for second Robot axis (visualized by 1_link)
    dh_params[4] = robot::DHParameters(0.0, 0.0, 0.35f, -3.1415f, 0.0, robot::REVOLUTE);  //
    dh_params[5] = robot::DHParameters(0.0, 0.0, 0.0, 1.5708f, 0.0, robot::REVOLUTE);  //
    dh_params[6] = robot::DHParameters(0.0, 0.0, 0.365f, -1.5708f, 0.0, robot::REVOLUTE);  //
    dh_params[7] = robot::DHParameters(0.0, 0.0, 0.0, 1.5708f, 0.0, robot::REVOLUTE);  //
    dh_params[8] = robot::DHParameters(0.0, 0.0, 0.0, 0.0, 0.0, robot::REVOLUTE);  // Params for last Robot axis (visualized by 6_link)
    dh_params[9] = robot::DHParameters(0.0, 0.0, 0.0, 0.0, 0.0, robot::REVOLUTE);  // Params for the not viusalized tool

    gvl->addRobot("myRobot", linknames, dh_params, paths_to_pointclouds, true);

    robot::JointValueMap min_joint_values;
    min_joint_values["z_translation"] = 0.0; // moves along the Z axis
    min_joint_values["y_translation"] = 1.0; // moves along the Y Axis
    min_joint_values["x_translation"] = 1.0; // moves along the X Axis
    min_joint_values["hollie/arm_0_link.xyz"] = -1.0;
    min_joint_values["hollie/arm_1_link.xyz"] = -1.0;
    min_joint_values["hollie/arm_2_link.xyz"] = -1.0;
    min_joint_values["hollie/arm_3_link.xyz"] = -1.0;
    min_joint_values["hollie/arm_4_link.xyz"] = -1.0;

    robot::JointValueMap max_joint_values;
    max_joint_values["z_translation"] = 0.0; // moves along the Z axis
    max_joint_values["y_translation"] = 1.0; // moves along the Y axis
    max_joint_values["x_translation"] = 1.0; // moves along the X Axis
    max_joint_values["hollie/arm_0_link.xyz"] = 1.5;
    max_joint_values["hollie/arm_1_link.xyz"] = 1.5;
    max_joint_values["hollie/arm_2_link.xyz"] = 1.5;
    max_joint_values["hollie/arm_3_link.xyz"] = 1.5;
    max_joint_values["hollie/arm_4_link.xyz"] = 1.5;


    // initialize the joint interpolation
    // not all joints have to be specified
    std::size_t counter = 0;
    /*
    constexpr int num_swept_volumes = 50;
    for (int i = 0; i < num_swept_volumes; ++i)
    {
        constexpr float ratio_delta = 0.02f;

        robot::JointValueMap myRobotJointValues = gpu_voxels::interpolateLinear(min_joint_values, max_joint_values,
	        ratio_delta * counter++);

        gvl->setRobotConfiguration("myRobot", myRobotJointValues);
        auto v = static_cast<BitVoxelMeaning>(eBVM_SWEPT_VOLUME_START + 1 + i);
        gvl->insertRobotIntoMap("myRobot", "myRobotMap", v);
    }*/
    gvl->visualizeMap(robot_id);

    constexpr float ratio_delta = 0.01f;
    for (int i = 0; i < 1024; ++i)
    {
        robot::JointValueMap myRobotJointValues = gpu_voxels::interpolateLinear(
            min_joint_values, max_joint_values, ratio_delta * counter++);

        gvl->setRobotConfiguration("myRobot", myRobotJointValues);

        gvl->insertRobotIntoMap("myRobot", robot_id, eBVM_OCCUPIED);
    }

    while (running)
    {
        /*
       * The robot moves and changes it's pose, so we "voxelize"
       * the links in every step and update the robot map.
       */
        LOGGING_INFO(Gpu_voxels, "Updating robot pose..." << endl);

        /*
        robot::JointValueMap myRobotJointValues = gpu_voxels::interpolateLinear(
            min_joint_values, max_joint_values, ratio_delta * counter++);

        gvl->setRobotConfiguration("myRobot", myRobotJointValues);

        gvl->insertRobotIntoMap("myRobot", robot_id, eBVM_OCCUPIED);*/

        /*
         * When the updates of the robot and the environment are
         * done, we can collide the maps and check for collisions.
         * The order of the maps is important here! The "smaller"
         * map should always be the first argument, as all occupied
         * Voxels from the first map will be looked up in the second map.
         * So, if you put a Voxelmap first, the GPU has to iterate over
         * the whole map in every step, to determine the occupied
         * Voxels. If you put an Octree first, the descend down to
         * the occupied Voxels is a lot more performant.
         */
        const auto robotMap = gvl->getMap(robot_id)->as<voxelmap::BitVectorVoxelMap>();
        const auto tmp = robotMap->as<voxelmap::BitVectorVoxelMap>();

        std::vector<BitVoxel<256>> buffer(tmp->getVoxelMapSize());
        cudaMemcpy(buffer.data(), robotMap->getConstVoidDeviceDataPtr(), robotMap->getMemoryUsage(), cudaMemcpyDeviceToHost);
        const auto dim = robotMap->getDimensions();
        auto sl = robotMap->getVoxelSideLength();
        auto sth = robotMap->getMetricDimensions();

        std::list<Vec3> out_raw;
        std::list<Vec3> out_filtered;
        //auto p_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        auto idx = [&dim](uint32_t x, uint32_t y, uint32_t z)
        {
            return z * dim.x * dim.y + y * dim.x + x;
        };
        auto occ_at = [&buffer, &idx](uint32_t x, uint32_t y, uint32_t z)
        {
            return buffer[idx(x, y, z)].isOccupied(0);
        };


        for (uint32_t z = 0; z < dim.z; ++z)
        {
	        for (uint32_t y = 0; y < dim.y; ++y)
	        {
		        for (uint32_t x = 0; x < dim.x; ++x)
		        {
                    if (occ_at(x, y, z))
                    {
                        out_raw.emplace_back(x, y, z);
                        //p_cloud->emplace_back((0.5 + x) * sl, (0.5 + y) * sl, (0.5 + z) * sl);

                        if (x == 0 || y == 0 || z == 0 || x == dim.x - 1 || y == dim.y - 1 || z == dim.z)
                            continue;

                        if (occ_at(x - 1, y, z) &&
                            occ_at(x + 1, y, z) &&
                            occ_at(x, y + 1, z) &&
                            occ_at(x, y - 1, z) &&
                            occ_at(x, y, z + 1) &&
                            occ_at(x, y, z - 1))
                        {
                            continue;
                        }
                        out_filtered.emplace_back(x, y, z);
                    }
		        }
	        }
        }
        /*
        for (auto it = out.begin(); it != out.end();)
        {
            if (it->x == 0 || it->y == 0 || it->z == 0)
                continue;

            if (out.contains({it->x - 1, it->y, it->z}) &&
                out.contains({ it->x + 1, it->y, it->z }) &&
                out.contains({ it->x, it->y + 1, it->z }) &&
                out.contains({ it->x, it->y - 1, it->z }) &&
                out.contains({ it->x, it->y, it->z + 1 }) &&
                out.contains({ it->x, it->y, it->z - 1}))
            {
                it = out.erase(it);
                continue;
            }
            ++it;
        }*/

        //pcl::io::savePLYFile("pcl.ply", *p_cloud);
        //return 0;
        /*
        for (const auto& sth : buffer | std::views::filter([](const BitVoxel<256>& v)
        {
        	return v.isOccupied(0.1f);
        }))
        {
            out.emplace_back(sth);
        }*/


        LOGGING_INFO(
            Gpu_voxels,
            "Collsions: " << gvl->getMap(env_id)->as<NTree::GvlNTreeDet>()->collideWith(robotMap) << endl);
        //"Collsions: " << gvl->getMap(env_id)->as<voxelmap::BitVectorVoxelMap>()->collideWith(robotMap) << endl);

    // visualize both maps
        gvl->visualizeMap(robot_id);
        gvl->visualizeMap(env_id);

        std::this_thread::sleep_for(std::chrono::microseconds(100000));

        // We assume that the robot will be updated in the next loop, so we clear the map.
        //gvl->clearMap(robot_id);
    }
    gvl.reset();
    return 0;
}
