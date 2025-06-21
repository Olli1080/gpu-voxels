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
#include <constants.hpp>

#include <icl_core_config/Config.h>

#include <gpu_voxels/GpuVoxels.h>
//#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

#include <mutex>
#include <ranges>
#include <set>

#include <pcl/point_cloud.h>
#include <pcl/impl/point_types.hpp>

#include <pcl/io/ply_io.h>

#include "gpu_voxels/helpers/MathHelpers.h"

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
    //const char* env_id = "myEnvironmentMap";

    float voxel_side_length = 0.01f;

    float down = 0.360f;
    float up = 1.190f;
    float vertical = down + up;

    float radius = 0.855f;

    int dim_xy = static_cast<int>(ceilf((2.f * radius) / voxel_side_length));
    int dim_z = static_cast<int>(ceilf(vertical / voxel_side_length));

    /*
     * First, we generate an API class, which defines the
     * volume of our space and the resolution.
     * Be careful here! The size is limited by the memory
     * of your GPU. Even if an empty Octree is small, a
     * Voxelmap will always require the full memory.
     */
    gvl = GpuVoxels::getInstance();
    gvl->initialize(dim_xy, dim_xy, dim_z, voxel_side_length);

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
    /*gvl->addMap(MT_BITVECTOR_OCTREE, env_id);

    if (!gvl->insertPointCloudFromFile(env_id, "pointcloud_0002.pcd", true,
        eBVM_OCCUPIED, true, Vector3f(-6.f, -7.3f, 0.f)))
    {
        LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
    }*/

    /*
     * Of course, we need a robot. At this point, you can choose between
     * describing your robot via ROS URDF or via conventional DH parameter.
     * In this example, we simply hardcode a DH robot:
     */

     // First, we load the robot geometry which contains 9 links with 7 geometries:
     // Geometries are required to have the same names as links, if they should get transformed.
    std::vector<std::string> linknames(8);
    std::vector<std::string> paths_to_pointclouds(8);
    //linknames[0] = "z_translation";
    //linknames[1] = "y_translation";
    //linknames[2] = "x_translation";

    linknames[0] = paths_to_pointclouds[0] = "franka/link0.binvox";
    linknames[1] = paths_to_pointclouds[1] = "franka/link1.binvox";
    linknames[2] = paths_to_pointclouds[2] = "franka/link2.binvox";
    linknames[3] = paths_to_pointclouds[3] = "franka/link3.binvox";
    linknames[4] = paths_to_pointclouds[4] = "franka/link4.binvox";
    linknames[5] = paths_to_pointclouds[5] = "franka/link5.binvox";
    linknames[6] = paths_to_pointclouds[6] = "franka/link6.binvox";
    linknames[7] = paths_to_pointclouds[7] = "franka/link7.binvox";

    std::vector<robot::DHParameters<robot::CRAIGS>> dh_params(8);
    // _d,  _theta,  _a,   _alpha, _value, _type
    //dh_params[0] = { 0.0, -M_PI_2, 0.0, -M_PI_2, 0.0, robot::PRISMATIC }; // Params for Y translation
    //dh_params[1] = { 0.0, -M_PI_2, 0.0, 3 * M_PI_2, 0.0, robot::PRISMATIC }; // Params for X translation
    //dh_params[2] = { 0.0, -M_PI_2, 0.0, -M_PI_2, 0, robot::PRISMATIC }; // Params for first Robot axis (visualized by 0_link)

    dh_params[0] = { 0.333f, 0.f, 0.0, 0.f, 0.0, robot::REVOLUTE };
    dh_params[1] = { 0.0, 0.0, 0.0, -M_PI_2, 0.0, robot::REVOLUTE };
    dh_params[2] = { 0.316f, 0.f, 0.f, M_PI_2, 0.0, robot::REVOLUTE };
    dh_params[3] = { 0.0, 0.f, 0.0825f, M_PI_2, 0.0, robot::REVOLUTE };
    dh_params[4] = { 0.384f, 0.0, -0.0825f, -M_PI_2, 0.0, robot::REVOLUTE };  //
    dh_params[5] = { 0.0, 0.0, 0.f, M_PI_2, 0.0, robot::REVOLUTE };  //
    dh_params[6] = { 0.0, 0.0, 0.088f, M_PI_2, 0.0, robot::REVOLUTE };
    dh_params[7] = { 0.107f, 0.0, 0.0, 0.f, 0.0, robot::REVOLUTE };  //

    gvl->addRobot("myRobot", linknames, dh_params, paths_to_pointclouds, true);

    robot::JointValueMap min_joint_values;
    //min_joint_values["z_translation"] = 1.0; // moves along the Z axis
    //min_joint_values["y_translation"] = 1.0; // moves along the Y Axis
    //min_joint_values["x_translation"] = 0.0; // moves along the X Axis
    min_joint_values["franka/link0.binvox"] = -2.8973f; // moves along the X Axis
    min_joint_values["franka/link1.binvox"] = -1.7628f;
    min_joint_values["franka/link2.binvox"] = -2.8973f;
    min_joint_values["franka/link3.binvox"] = -3.0718f;
    min_joint_values["franka/link4.binvox"] = -2.8973f;
    min_joint_values["franka/link5.binvox"] = -0.0175f;
    min_joint_values["franka/link6.binvox"] = -2.8973f;
    min_joint_values["franka/link7.binvox"] = 0.f;

    robot::JointValueMap max_joint_values;
    //max_joint_values["z_translation"] = 1.0; // moves along the Z axis
    //max_joint_values["y_translation"] = 1.0; // moves along the Y axis
    //max_joint_values["x_translation"] = 0.0; // moves along the X axis
    max_joint_values["franka/link0.binvox"] = 2.8973f; // moves along the X axis
    max_joint_values["franka/link1.binvox"] = 1.7628f;
    max_joint_values["franka/link2.binvox"] = 2.8973f;
    max_joint_values["franka/link3.binvox"] = -0.0698f;
    max_joint_values["franka/link4.binvox"] = 2.8973f;
    max_joint_values["franka/link5.binvox"] = 3.7525f;
    max_joint_values["franka/link6.binvox"] = 2.8973f;
    max_joint_values["franka/link7.binvox"] = 0.f;
    /*
    Matrix4f out = Matrix4f::createIdentity();
    for (int i = 0; i < 3; ++i)
    {
        Matrix4f temp;
        auto cpy = dh_params[i];
        if (i < 2)
            cpy.value = 1.f;
        cpy.convertDHtoM(temp);
        std::cout << temp << std::endl;
        out = out * temp;
    }
    std::cout << out << std::endl;
    std::cout << "--------------------------" << std::endl;

    dh_params[3].convertDHtoM(out);
    std::cout << out << std::endl;

    out = Matrix4f::createIdentity();
    for (int i = 0; i < 4; ++i)
    {
        Matrix4f temp;
        auto cpy = dh_params[i];
        if (i < 2)
            cpy.value = 1.f;
        cpy.convertDHtoM(temp);
        out = out * temp;
    }
    std::cout << out << std::endl;
    */
    /*std::vector<robot::DHParameters<robot::CLASSIC>> dh_params2(3);
    dh_params2[0] = { 0.0, 0.0, 0.0, -M_PI_2, 0.0, robot::PRISMATIC }; // Params for Y translation
    dh_params2[1] = { 0.0, -M_PI_2, 0.0, -M_PI_2, 0.0, robot::PRISMATIC }; // Params for X translation
    dh_params2[2] = { 0.0, M_PI_2, 0.0, M_PI_2, 0.0, robot::PRISMATIC }; // Params for first Robot axis (visualized by 0_link)

    out = Matrix4f::createIdentity();
    for (int i = 0; i < 3; ++i)
    {
        Matrix4f temp;
        auto cpy = dh_params2[i];
        if (i != 0)
            cpy.value = 1.f;
        cpy.convertDHtoM(temp);
        out = out * temp;
    }
    std::cout << out << std::endl;
    */
    /*
    for (int i = -4; i < 9; ++i)
    {
        for (int j = -4; j < 9; ++j)
        {
            for (int k = -4; k < 9; ++k)
            {
                for (int l = -4; l < 9; ++l)
                {
                    for (int m = -4; m < 9; ++m)
                    {
                        for (int n = -4; n < 9; ++n)
                        {
                            std::vector<robot::DHParameters<robot::CRAIGS>> dh_params(3);
                            dh_params[0] = { 0.0, float(i * M_PI_4), 0.0, float(j * M_PI_4), 0.0, robot::PRISMATIC }; // Params for Y translation
                            dh_params[1] = { 0.0, float(k * M_PI_4), 0.0, float(l * M_PI_4), 0.0, robot::PRISMATIC }; // Params for X translation
                            dh_params[2] = { 0.0, float(m * M_PI_4), 0.0, float(n * M_PI_4), 0.0, robot::PRISMATIC }; // Params for first Robot axis (visualized by 0_link)
                            //dh_params[3] = { 0.333f, 0.f, 0.0, 0.f, 0.0, robot::REVOLUTE };
                            
                            {
                                Matrix4f out = Matrix4f::createIdentity();
                                for (int i = 0; i < 3; ++i)
                                {
                                    Matrix4f temp;
                                    auto cpy = dh_params[i];
                                    if (i != 0 && i < 3)
                                        cpy.value = 1.f;
                                    cpy.convertDHtoM(temp);
                                    out = out * temp;
                                }
                                if (out.a14 == 1.f && out.a24 == 1.f && out.a33 == 1.f && out.a11 == 1.f && out.a22 == 1.f)
                                {
                                    for (int i = 0; i < 3; ++i)
                                        std::cout << dh_params[i] << std::endl;
                                    std::cout << std::endl << "i != " << 0 << std::endl;
                                    std::cout << out << std::endl;
                                }
                            }
                            {
                                Matrix4f out = Matrix4f::createIdentity();
                                for (int i = 0; i < 3; ++i)
                                {
                                    Matrix4f temp;
                                    auto cpy = dh_params[i];
                                    if (i != 1 && i < 3)
                                        cpy.value = 1.f;
                                    cpy.convertDHtoM(temp);
                                    out = out * temp;
                                }
                                //(out.a14 == 1.f && out.a24 == 1.f && out.a33 == 1.f && out.a11 == 1.f && out.a22 == 1.f)
                                //(out.a21 == -1.f && out.a12 == 1.f && out.a33 == 1.f && out.a14 == 1.f && out.a24 == 1.f)
                                if (out.a14 == 1.f && out.a24 == 1.f && out.a33 == 1.f && out.a11 == 1.f && out.a22 == 1.f)
                                {
                                    for (int i = 0; i < 3; ++i)
                                        std::cout << dh_params[i] << std::endl;
                                    std::cout << std::endl << "i != " << 1 << std::endl;
                                    std::cout << out << std::endl;
                                }
                            }
                            {
                                Matrix4f out3 = Matrix4f::createIdentity();
                                for (int iu = 0; iu < 3; ++iu)
                                {
                                    auto cpy = dh_params[iu];
                                    if (iu != 2 && iu < 3)
                                        cpy.value = 1.f;

                                    Matrix4f temp;
                                    cpy.convertDHtoM(temp);
                                    out3 = out3 * temp;
                                }
                                if (out3.a14 == 1.f && out3.a24 == 1.f && out3.a33 == 1.f && out3.a11 == 1.f && out3.a22 == 1.f)
                                {
                                    for (int iu = 0; iu < 3; ++iu)
                                    {
                                        std::cout << dh_params[iu] << std::endl;
                                        Matrix4f mat;
                                        if (iu < 2)
                                            dh_params[iu].value = 1.f;
                                        dh_params[iu].convertDHtoM(mat);
                                        std::cout << mat << std::endl;
                                    }

                                    std::cout << std::endl << "i != " << 2 << std::endl;
                                    std::cout << out3 << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }*/
    
    // initialize the joint interpolation
    // not all joints have to be specified
    std::size_t counter = 0;
    constexpr float ratio_delta = 0.02f;
    /*
    constexpr int num_swept_volumes = 50;
    for (int i = 0; i < num_swept_volumes; ++i)
    {
        

        robot::JointValueMap myRobotJointValues = gpu_voxels::interpolateLinear(min_joint_values, max_joint_values,
	        ratio_delta * counter++);

        gvl->setRobotConfiguration("myRobot", myRobotJointValues);
        auto v = static_cast<BitVoxelMeaning>(eBVM_SWEPT_VOLUME_START + 1 + i);
        gvl->insertRobotIntoMap("myRobot", "myRobotMap", v);
    }*/
    gvl->visualizeMap(robot_id);

    gvl->setRobotBaseTransformation("myRobot", Eigen::Affine3f(Eigen::Translation3f(radius, radius, down)).matrix());

    /*constexpr float ratio_delta = 0.01f;
    for (int i = 0; i < 1; ++i)
    {
        //robot::JointValueMap myRobotJointValues = gpu_voxels::interpolateLinear(
          //  min_joint_values, max_joint_values, ratio_delta * counter++);

        robot::JointValueMap myRobotJointValues = {
            //{ "z_translation", 1.f },
            //{ "y_translation", 1.f },
            //{ "x_translation", 0.f },
            { "franka/link0.binvox", 0.f },
            { "franka/link1.binvox", 0.f },
            { "franka/link2.binvox", 0.f },
            { "franka/link3.binvox", 0.f },
            { "franka/link4.binvox", 0.f },
            { "franka/link5.binvox", 0.f },
            { "franka/link6.binvox", 0.f },
            { "franka/link7.binvox", 0.f }
        };



        constexpr size_t itm_points = 3;

        for (int i = 0; i <= itm_points; ++i)
            for (int j = 0; j <= itm_points; ++j)
                for (int k = 0; k <= itm_points; ++k)
                    for (int l = 0; l <= itm_points; ++l)
                        for (int m = 0; m <= itm_points; ++m)
                            for (int n = 0; n <= itm_points; ++n)
                                for (int o = 0; o <= itm_points; ++o)
                                {
                                    auto config = gpu_voxels::interpolateLinear(min_joint_values, max_joint_values,
                                        robot::JointValueMap{
                                            { "franka/link0.binvox", static_cast<float>(i) / itm_points },
                                            { "franka/link1.binvox", static_cast<float>(j) / itm_points },
                                            { "franka/link2.binvox", static_cast<float>(k) / itm_points },
                                            { "franka/link3.binvox", static_cast<float>(l) / itm_points },
                                            { "franka/link4.binvox", static_cast<float>(m) / itm_points },
                                            { "franka/link5.binvox", static_cast<float>(n) / itm_points },
                                            { "franka/link6.binvox", static_cast<float>(o) / itm_points },
                                            { "franka/link7.binvox", 0.f }
                                        });

                                    gvl->setRobotConfiguration("myRobot", config);
                                    gvl->insertRobotIntoMap("myRobot", robot_id, eBVM_OCCUPIED);
                                }
                                    


        //gvl->setRobotConfiguration("myRobot", myRobotJointValues);
        

        //gvl->insertRobotIntoMap("myRobot", robot_id, eBVM_OCCUPIED);
    }*/

    while (running)
    {
        auto t0 = std::chrono::steady_clock::now();
        /*
       * The robot moves and changes it's pose, so we "voxelize"
       * the links in every step and update the robot map.
       */
        LOGGING_INFO(Gpu_voxels, "Updating robot pose..." << endl);

        
        robot::JointValueMap myRobotJointValues = gpu_voxels::interpolateLinear(
            min_joint_values, max_joint_values, ratio_delta * counter++);

        gvl->setRobotConfiguration("myRobot", myRobotJointValues);

        gvl->insertRobotIntoMap("myRobot", robot_id, eBVM_OCCUPIED);

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
        //const auto tmp = robotMap->as<voxelmap::BitVectorVoxelMap>();

        //thrust::host_vector<BitVoxel<BIT_VECTOR_LENGTH>> buffer = dev_data;

        const auto dim = robotMap->getDimensions();
        auto sl = robotMap->getVoxelSideLength();
        auto sth = robotMap->getMetricDimensions();

        //std::list<Vec3> out_raw;
        //std::list<Vec3> out_filtered;
        //auto p_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        /*auto idx = [&dim](uint32_t x, uint32_t y, uint32_t z)
        {
            return z * dim.x() * dim.y() + y * dim.x() + x;
        };
        auto occ_at = [&buffer, &idx](uint32_t x, uint32_t y, uint32_t z)
        {
            return buffer[idx(x, y, z)].isOccupied(0);
        };*/

        const auto& dev_data = robotMap->getDeviceData();
        auto final_result = voxelmap::extract_visual_voxels(dev_data, dim);

        /*
        for (uint32_t z = 0; z < dim.z(); ++z)
        {
	        for (uint32_t y = 0; y < dim.y(); ++y)
	        {
		        for (uint32_t x = 0; x < dim.x(); ++x)
		        {
                    if (occ_at(x, y, z))
                    {
                        out_raw.emplace_back(x, y, z);
                        //p_cloud->emplace_back((0.5 + x) * sl, (0.5 + y) * sl, (0.5 + z) * sl);

                        if (x > 0 && occ_at(x - 1, y, z) &&
                            x < dim.x() - 2 && occ_at(x + 1, y, z) &&
                            y > 0 && occ_at(x, y - 1, z) &&
                            y < dim.y() - 2 && occ_at(x, y + 1, z) &&
                            z > 0 && occ_at(x, y, z - 1) &&
                            z < dim.z() - 2 && occ_at(x, y, z + 1))
                            continue;

                        out_filtered.emplace_back(x, y, z);
                    }
		        }
	        }
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


        /*LOGGING_INFO(
            Gpu_voxels,
            "Collsions: " << gvl->getMap(env_id)->as<NTree::GvlNTreeDet>()->collideWith(robotMap) << endl);*/
        //"Collsions: " << gvl->getMap(env_id)->as<voxelmap::BitVectorVoxelMap>()->collideWith(robotMap) << endl);

		// visualize both maps
        gvl->visualizeMap(robot_id);
        //gvl->visualizeMap(env_id);

        std::this_thread::sleep_for(std::chrono::microseconds(100000));

        std::cout << "Time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0) << " seconds" << std::endl;
        // We assume that the robot will be updated in the next loop, so we clear the map.
        //gvl->clearMap(robot_id);
    }
    gvl.reset();
    return 0;
}
