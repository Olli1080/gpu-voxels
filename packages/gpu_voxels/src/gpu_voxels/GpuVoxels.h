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


// This is used for Doxygens index page:
/*!
 * \mainpage GPU-Voxels
 * \htmlinclude gvl_doxygen_intro.html
 */

 //----------------------------------------------------------------------
 /*!\file
  *
  * \author  Andreas Hermann
  * \date    2014-06-08
  *
  * This high level API offers a lot of convenience functionality
  * and additional sanity checks. If you prefer a streamlined high
  * performance API, you can manage the shared pointers to maps and
  * robots by yourself and directly work with them.
  *
  */
  //----------------------------------------------------------------------
#ifndef GPU_VOXELS_GPU_VOXELS_H_INCLUDED
#define GPU_VOXELS_GPU_VOXELS_H_INCLUDED

#include <memory>
#include <map>

#include <gpu_voxels/GpuVoxelsMap.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/octree/Octree.h>
#include <gpu_voxels/voxellist/VoxelList.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>

#include <gpu_voxels/robot/robot_interface.h>

#ifdef _IC_BUILDER_GPU_VOXELS_URDF_ROBOT_
#include <gpu_voxels/robot/urdf_robot/urdf_robot.h>
#endif

#include <gpu_voxels/robot/dh_robot/KinematicChain.h>

#include <gpu_voxels/logging/logging_gpu_voxels.h>

/**
 * @namespace gpu_voxels
 * Library for GPU based Voxel Collision Detection
 */
namespace gpu_voxels {

	struct ManagedMap;
	struct ManagedPrimitiveArray;

	class VisProvider;
	class GpuVoxels;

	typedef std::shared_ptr<gpu_voxels::GpuVoxels> GpuVoxelsSharedPtr;

	typedef std::shared_ptr<cudaIpcMemHandle_t> CudaIpcMemHandleSharedPtr;
	typedef std::map<std::string, ManagedMap> ManagedMaps;
	typedef ManagedMaps::iterator ManagedMapsIterator;

	typedef std::map<std::string, ManagedPrimitiveArray> ManagedPrimitiveArrays;
	typedef ManagedPrimitiveArrays::iterator ManagedPrimitiveArraysIterator;

	typedef std::shared_ptr<robot::RobotInterface> RobotInterfaceSharedPtr;
	typedef std::map<std::string, RobotInterfaceSharedPtr> ManagedRobots;
	typedef ManagedRobots::iterator ManagedRobotsIterator;
	
	template<std::size_t num_bits>
	class BitVector;


	namespace primitive_array
	{
		enum PrimitiveType : uint8_t;
	}

	class GpuVoxels
	{
	public:

		~GpuVoxels();

		/*!
		 * \brief getInstance creates a Singleton object of GpuVoxels
		 * in C++11 this method is thread safe. See the C++11 Standard "Chapter 6.7 Declaration Statement"
		 * for more information. On multiprocessor system this method might not be thread safe.
		 * \return pointer to the singleton object
		 */
		static GpuVoxelsSharedPtr getInstance();

		/*!
		 * \brief initialize
		 * \param dim_x The map's x dimension
		 * \param dim_y The map's y dimension
		 * \param dim_z The map's z dimension
		 * \param voxel_side_length Defines the maximum resolution
		 */
		void initialize(uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float voxel_side_length);

		/*!
		 * \brief addMap Add a new map to GVL.
		 * \param map_type Choose between a representation: Octree, Voxelmap,
		 * Voxellist are possible
		 * \param map_name The name of the map for later identification
		 * \return Returns shared_ptr to the added map if adding was successful, otherwise returns empty shared_ptr
		 */
		GpuVoxelsMapSharedPtr addMap(MapType map_type, const std::string& map_name);

		/*!
		 * \brief delMap Remove a map from GVL.
		 * \param map_name Name of the map, that should be deleted.
		 * \return Returns true, if deleting was successful, false otherwise
		 */
		bool delMap(const std::string& map_name);

		/*!
		 * \brief clearMap Deletes ALL data from the map
		 * \param map_name Which map to clear
		 */
		bool clearMap(const std::string& map_name);

		/*!
		 * \brief clearMap Deletes a special voxel type from the map
		 * \param map_name Which map to clear
		 * \param voxel_meaning Which type of voxels to clear
		 */
		bool clearMap(const std::string& map_name, BitVoxelMeaning voxel_meaning);

		/*!
		 * \brief getMap Gets a const pointer to the map.
		 * \param map_name The name of the queried map.
		 * \return Pointer to the queried map. nullptr if map was not found.
		 */
		GpuVoxelsMapSharedPtr getMap(const std::string& map_name);

		/*!
		 * \brief visualizeMap Visualizes the map only if necessary.
		 * That's the case if it's enforced by \code force_repaint = true
		 * or the visualizer requested it.
		 * \param map_name Name of the map, that should be visualized.
		 * \param force_repaint True to force a repainting of the map. e.g. needed
		 * to visualize changed map data
		 * \return Returns true, if there was work to do, false otherwise.
		 */
		bool visualizeMap(const std::string& map_name, bool force_repaint = true);

		/*!
		 * \brief visualizePrimitivesArray Visualizes the array of primitives only if necessary.
		 * That's the case if it's enforced by \code force_repaint = true or the visualizer requested it.
		 * \param prim_array_name Name of the array, that should be visualized.
		 * \param force_repaint True to force a repainting of the array. e.g. needed to visualize changed map data
		 * \return Returns true, if there was work to do, false otherwise.
		 */
		bool visualizePrimitivesArray(const std::string& prim_array_name, bool force_repaint = true);

		/*!
		 * \brief addRobot Define a robot with its geometries and kinematic structure via DH parameter.
		 * Important: \code link_names have to be the same as \code paths_to_pointclouds if the pointclouds
		 * should get transformed by the kinematic!
		 * \param robot_name Name of the robot, used as handler
		 * \param link_names Vector of unique names of the rigid bodies of all links.
		 * \param dh_params DH representation of the robots kinematics.
		 * Has to be of the same dimensionality as the \code robot_cloud
		 * \param paths_to_pointclouds Files on disk that hold the pointcloud representation of the robot geometry
		 * \param use_model_path Search pointcloud files in directory specified by GPU_VOXELS_MODEL_PATH environment variable
		 * \return true, if robot was added, false otherwise
		 */
		template<robot::DHConvention convention>
		bool addRobot(const std::string& robot_name, const std::vector<std::string>& link_names,
			const std::vector<robot::DHParameters<convention>>& dh_params,
			const std::vector<std::string>& paths_to_pointclouds, bool use_model_path)
		{
			// check if robot with same name already exists
			const auto it = m_managed_robots.find(robot_name);
			if (it != m_managed_robots.end())
			{
				LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Robot with name '" << robot_name << "' already exists." << endl);
				return false;
			}

			m_managed_robots.emplace(robot_name, RobotInterfaceSharedPtr(
				new robot::KinematicChain<convention>(link_names, dh_params, paths_to_pointclouds, use_model_path)));

			return true;
		}


		/*!
		 * \brief addRobot Define a robot with its geometries and kinematic structure via DH parameter.
		 * Important: \code link_names have to be the same as \code paths_to_pointclouds if the pointclouds
		 * should get transformed by the kinematic!
		 * \param robot_name Name of the robot, used as handler
		 * \param link_names Vector of unique names of the rigid bodies of all links.
		 * \param dh_params DH representation of the robots kinematics.
		 * \param pointclouds Already existing \code MetaPointCloud of the robot's links with matching \code link_names
		 * \return true, if robot was added, false otherwise
		 */
		template<robot::DHConvention convention>
		bool addRobot(const std::string& robot_name, const std::vector<std::string>& link_names,
			const std::vector<robot::DHParameters<convention>>& dh_params,
			const MetaPointCloud& pointclouds)
		{
			// check if robot with same name already exists
			const auto it = m_managed_robots.find(robot_name);
			if (it != m_managed_robots.end())
			{
				LOGGING_ERROR_C(Gpu_voxels, GpuVoxels, "Robot with name '" << robot_name << "' already exists." << endl);
				return false;
			}

			m_managed_robots.emplace(robot_name, std::static_pointer_cast<robot::RobotInterface>(
				std::make_shared<robot::KinematicChain<convention>>(link_names, dh_params, pointclouds)));

			return true;
		}

#ifdef _BUILD_GVL_WITH_URDF_SUPPORT_
		/*!
		 * \brief addRobot Define a robot with its geometries and kinematic structure via a ROS URDF file.
		 * During parsing all meshses get replaced by pointclouds with the same name.
		 * \param robot_name Name of the robot, used as handler
		 * \param path_to_urdf_file Path to the URDF to load.
		 * \param use_model_path Search URDF file in path specified in GPU_VOXELS_MODEL_PATH environment variable
		 * \return true, if robot was added, false otherwise
		 */
		bool addRobot(const std::string& robot_name, const std::string& path_to_urdf_file, const bool use_model_path);
#endif

		/*!
		 * \brief getRobot Returns the shared pointer to the robot
		 * \param rob_name Name of the robot to get.
		 * \return Empty pointer, if not found.
		 */
		RobotInterfaceSharedPtr getRobot(const std::string& rob_name);

		/*!
		 * \brief setRobotConfiguration Changes the robot joint configuration and triggers the transformation
		 * of all joint's pointclouds. Call \code insertRobotIntoMap() afterwards!
		 * \param jointmap Map of jointnames and values. Not required to contain all joints of the robot.
		 * \return true, if update was successful
		 */
		bool setRobotConfiguration(const std::string& robot_name, const robot::JointValueMap& jointmap);

		bool setRobotBaseTransformation(const std::string& robot_name, const Matrix4f& transformation);

		/*!
		 * \brief updateRobotPart Changes the geometry of a single robot link. This is useful when changing a tool,
		 * grasping an object of when interpreting sensor data from an onboard sensor as a robot link.
		 * Caution: This function requires intensive memory access, if the size of the pointcloud changes!
		 * Call \code insertRobotIntoMap() afterwards!
		 * \param robot_name Name of the robot beeing modified
		 * \param link Index of the link that is modified
		 * \param pointcloud New pointcloud of the link. May differ in size. In that case, the function has higher runtime.
		 * \return true, if robot was modified, false otherwise
		 */
		bool updateRobotPart(const std::string& robot_name, const std::string& link_name, const std::vector<Vector3f>&
			pointcloud);

		/**
		 * @brief getRobotConfiguration Query the current configuration of a robot
		 * @param robot_name The robot's identifier
		 * @param jointmap Map with joint values. Missing joints will be added to map.
		 * @return True if robot with given identifier exists, false otherwise.
		 */
		bool getRobotConfiguration(const std::string& robot_name, robot::JointValueMap& jointmap);

		/*!
		 * \brief insertPointCloudFromFile inserts a pointcloud from a file into the map
		 * The coordinates are interpreted as global coordinates
		 * \param map_name Name of the map to insert the pointcloud
		 * \param path filename (Must end in .xyz for XYZ files, .pcd for PCD files or .binvox for Binvox files)
		 * \param use_model_path Prepends environment variable GPU_VOXELS_MODEL_PATH to path if true
		 * \param shift_to_zero if true, the map will be shifted, so that its minimum lies at zero.
		 * \param offset_XYZ if given, the map will be transformed by this XYZ offset. If shifting is active, this happens after the shifting.
		 * \return true if succeeded, false otherwise
		 */
		bool insertPointCloudFromFile(const std::string& map_name, const std::string& path, bool use_model_path,
			BitVoxelMeaning voxel_meaning, bool shift_to_zero = false,
			const Vector3f& offset_XYZ = Vector3f::Zero(), float scaling = 1.f);

		/*!
		 * @brief insertPointCloudIntoMap Inserts a PointCloud into the map.
		 * @param cloud The PointCloud to insert
		 * @param voxel_meaning Voxel meaning of all voxels
		 */
		bool insertPointCloudIntoMap(const PointCloud& cloud, const std::string& map_name,
			BitVoxelMeaning voxel_meaning);

		/*!
		 * @brief insertPointCloudIntoMap Inserts a PointCloud into the map.
		 * @param cloud The PointCloud to insert
		 * @param voxel_meaning Voxel meaning of all voxels
		 */
		bool insertPointCloudIntoMap(const std::vector<Vector3f>& cloud, const std::string& map_name,
			BitVoxelMeaning voxel_meaning);

		/*!
		 * @brief insertMetaPointCloudIntoMap Inserts a MetaPointCloud into the map. Each pointcloud
		 * inside the MetaPointCloud will get it's own voxel meaning as given in the voxel_meanings
		 * parameter. The number of pointclouds in the MetaPointCloud and the size of voxel_meanings
		 * have to be identical.
		 * @param meta_point_cloud The MetaPointCloud to insert
		 * @param voxel_meanings Vector with voxel meanings of sub clouds
		 */
		bool insertMetaPointCloudIntoMap(const MetaPointCloud& meta_point_cloud,
			const std::string& map_name,
			const std::vector<BitVoxelMeaning>& voxel_meanings);

		/*!
		 * @brief insertMetaPointCloudIntoMap Inserts a MetaPointCloud into the map.
		 * @param meta_point_cloud The MetaPointCloud to insert
		 * @param voxel_meaning Voxel meaning of all sub-clouds voxels
		 */
		bool insertMetaPointCloudIntoMap(const MetaPointCloud& meta_point_cloud,
			const std::string& map_name,
			BitVoxelMeaning voxel_meaning);


		/*!
		 * \brief insertRobotIntoMap Writes a robot with its current pose into a map
		 * \param robot_name Name of the robot to use
		 * \param map_name Name of the map to insert the robot
		 * \return true, if robot was added, false otherwise
		 */
		bool insertRobotIntoMap(const std::string& robot_name, const std::string& map_name, BitVoxelMeaning voxel_meaning);


		/*!
		 * \brief insertRobotIntoMapSelfCollAware This inserts a robot and checks for every link, if it collides with previously inserted data.
		 * For performance reasons it is advised to also give optional parameters to spare allocations at every call.
		 * \param robot_name Name of the robot to use
		 * \param map_name Name of the map to insert the robot
		 * \param voxel_meanings Can be used to define the meanings of the pointclouds. Useful, if more than one robot is inserted into the map.
		 * If not given, meanings from eBVM_SWEPT_VOLUME_START on are used and incremented per link. Requires one entry per robot subcloud.
		 * \param collision_masks Can be used to mask out collision pairs. Requires one entry per robot subcloud. If bits are set, collisions will be checked.
		 * \param colliding_meanings Can be used to get bitvector with collision results. Same meanings as given in \c voxel_meanings.
		 * \return True, if a collision with previously inserted data occurred.
		 */
		bool insertRobotIntoMapSelfCollAware(const std::string& robot_name, const std::string& map_name,
			const std::vector<BitVoxelMeaning>& voxel_meanings = {},
			const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks = {},
			BitVector<BIT_VECTOR_LENGTH>* colliding_meanings = nullptr);

		/*!
		* \brief insertBoxIntoMap Helper function to generate obstacles. This inserts a box object.
		* \param corner_min Coordinates of the lower, left corner in the front.
		* \param corner_max Coordinates of the upper, right corner in the back.
		* \param map_name Name of the map to insert the box
		* \param voxel_meaning The kind of voxel to insert
		* \param points_per_voxel Point density. This is only relevant to test probabilistic maps.
		*/
		bool insertBoxIntoMap(const Vector3f& corner_min, const Vector3f& corner_max, const std::string& map_name, BitVoxelMeaning voxel_meaning, uint16_t points_per_voxel = 1);

		/*!
		 * \brief addPrimitives
		 * \param prim_type Cubes or Spheres
		 * \param array_name Name of the new array
		 * \return true if successful, false otherwise
		 */
		bool addPrimitives(primitive_array::PrimitiveType prim_type, const std::string& array_name);

		/*!
		 * \brief delPrimitives
		 * \param array_name Name of the array to delete
		 * \return true if successful, false otherwise
		 */
		bool delPrimitives(const std::string& array_name);

		/*!
		 * \brief modifyPrimitives Creates or updates points and sizes of the primitives.
		 * \param array_name Name of array to modify
		 * \param prim_positions Vector of new metric positions / metric sizes.
		 * \return true if successful, false otherwise
		 */
		bool modifyPrimitives(const std::string& array_name, const std::vector<Vector4f>& prim_positions);

		/*!
		 * \brief modifyPrimitives Creates or updates points and sizes of the primitives.
		 * \param array_name Name of array to modify
		 * \param prim_positions Vector of new positions / sizes. Given in Voxels, not metric!
		 * \return true if successful, false otherwise
		 */
		bool modifyPrimitives(const std::string& array_name, const std::vector<Vector4i>& prim_positions);

		/*!
		 * \brief modifyPrimitives Creates or updates points the primitives in the array. All have equal diameter.
		 * \param array_name Name of array to modify
		 * \param prim_positions Vector of new metric positions
		 * \param diameter The metric diameter of all primitives
		 * \return true if successful, false otherwise
		 */
		bool modifyPrimitives(const std::string& array_name, const std::vector<Vector3f>& prim_positions, const float& diameter);

		/*!
		 * \brief modifyPrimitives Creates or updates points that represent primitives. All have equal diameter.
		 * \param array_name Name of array to modify
		 * \param prim_positions Vector of new positions. Given in Voxels, not metric!
		 * \param diameter The diameter of all primitives. Given in Voxels, not metric!
		 * \return true if successful, false otherwise
		 */
		bool modifyPrimitives(const std::string& array_name, const std::vector<Vector3i>& prim_positions, const uint32_t& diameter);
		/*!
		 * \brief getVisualization Gets a handle to the visualization interface of this map.
		 * \return pointer to \code VisProvider of the map with the given name.
		 */
		VisProvider* getVisualization(const std::string& map_name);


		/**
		 * @brief Gets the dimensions of voxel space
		 *
		 * @param dim_x [out] number of voxels in x_dimension
		 * @param dim_y [out] number of voxels in y_dimension
		 * @param dim_z [out] number of voxels in z_dimension
		 * @return void
		 */
		void getDimensions(uint32_t& dim_x, uint32_t& dim_y, uint32_t& dim_z) const;

		/**
		 * @brief Gets the dimensions of voxel space
		 * @param dim [out] number of Voxels in each dimension
		 * @return void
		 */
		void getDimensions(Vector3ui& dim) const;

		/**
		 * @brief getVoxelSideLength Gets the sidelength of voxels
		 * @param voxel_side_length [out] sidelength of voxels
		 */
		void getVoxelSideLength(float& voxel_side_length) const;


	protected:

		/*!
		 * \brief gvl Constructor, to define the general resolution and size of the represented volume
		 * it is necessary to call the initialize method.
		 * This is relevant for the VoxelMap / VoxelList.
		 * The Octree depth will be chosen accordingly.
		 */
		GpuVoxels();

	private:

		static std::weak_ptr<GpuVoxels> masterPtr;

		ManagedMaps m_managed_maps;
		ManagedRobots m_managed_robots;
		ManagedPrimitiveArrays m_managed_primitive_arrays;
		gpu_voxels::Vector3ui m_dim;
		float m_voxel_side_length;
	};

} // end of namespace
#endif