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
 * \date    2012-09-13
 *
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_ROBOT_DH_ROBOT_KINEMATIC_CHAIN_H_INCLUDED
#define GPU_VOXELS_ROBOT_DH_ROBOT_KINEMATIC_CHAIN_H_INCLUDED

#include <vector>
#include <string>
#include <map>

#include <gpu_voxels/robot/robot_interface.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/robot/dh_robot/KinematicLink.h>

namespace gpu_voxels
{
	namespace robot
	{
		template<DHConvention convention>
		class KinematicChain : public RobotInterface
		{
		public:

			/*!
			 * \brief KinematicChain constructor that loads pointclouds from files
			 * \param linknames Names of all links
			 * \param dh_params DH Parameters of all links
			 * \param paths_to_pointclouds Filepaths to pointclouds
			 * \param use_model_path Use GPU_MODEL_PATH to search for pointclouds
			 *
			 * Important: Linknames have to be the same as Pointcloud names
			 * (derived from paths_to_pointclouds), if they should be kinematically
			 * transformed.
			 */
			__host__
			KinematicChain(const std::vector<std::string>& linknames,
				const std::vector<robot::DHParameters<convention>>& dh_params,
				const std::vector<std::string>& paths_to_pointclouds,
				const std::filesystem::path& model_path,
				const Matrix4f& base_transformation = Matrix4f::Identity());

			/*!
			 * \brief KinematicChain constructor that takes existing pointcloud
			 * \param linknames Names of all links
			 * \param dh_params DH Parameters of all links
			 * \param pointclouds Existing Meta-Pointcloud of robt
			 * (e.g. including attached Sensor Pointcloud)
			 *
			 * Important: Linknames have to be the same as Pointcloud names
			 * (derived from paths_to_pointclouds), if they should be kinematically
			 * transformed.
			 */
			__host__
			KinematicChain(const std::vector<std::string>& linknames,
				const std::vector<robot::DHParameters<convention>>& dh_params,
				const MetaPointCloud& pointclouds,
				const Matrix4f& base_transformation = Matrix4f::Identity());

			__host__
			~KinematicChain() override = default;

			/**
			 * @brief getJointNames Reads all joint names
			 * @param jointnames Vector of jointnames that will get extended
			 */
			void getJointNames(std::vector<std::string>& jointnames) override;

			/*!
			 * \brief setConfiguration Sets a robot configuration
			 * and triggers pointcloud transformation.
			 * \param joint_values Robot joint values. Will get
			 * matched by names, so not all joints have to be specified.
			 */
			__host__
			void setConfiguration(const JointValueMap& joint_values) override;

			/*!
			 * \brief setConfiguration Reads the current config
			 * \param joint_values This map will get extended, if
			 * jointnames are missing.
			 */
			__host__
			void getConfiguration(JointValueMap& joint_values) override;


			/**
			 * @brief getLowerJointLimits Gets the minimum joint values
			 * @param lower_limits Map of jointnames and values.
			 * This map will get extended if joints were missing.
			 */
			void getLowerJointLimits(JointValueMap& lower_limits) override;

			/**
			 * @brief getUpperJointLimits Gets the maximum joint values
			 * @param upper_limits Map of jointnames and values.
			 * This map will get extended if joints were missing.
			 */
			void getUpperJointLimits(JointValueMap& upper_limits) override;

			/**
			 * @brief getTransformedClouds
			 * @return Pointers to the kinematically transformed clouds.
			 */
			const MetaPointCloud* getTransformedClouds() override;

			/*!
			 * \brief updatePointcloud Changes the geometry of a single link.
			 * Useful when grasping an object, changing a tool
			 * or interpreting point cloud data from an onboard sensor as a robot link.
			 * \param link Link to modify
			 * \param cloud New geometry
			 */
			void updatePointcloud(const std::string& link_name, const std::vector<Vector3f>& cloud) override;


			void setBaseTransformation(const Matrix4f& base_transformation) override;

			void getBaseTransformation(Matrix4f& base_transformation) const override;

			[[nodiscard]] Matrix4f getTransform(size_t idx) const override;
			//[[nodiscard]] Vector3f transform_point(const Eigen::Vector3f& p) const override;

			//! for testing purposes
			//__host__
			//void transformPointAlongChain(Vector3f point);

		private:

			void init(const std::vector<std::string>& linknames,
				const std::vector<robot::DHParameters<convention>>& dh_params,
				const Matrix4f& base_transformation
			);

			void performPointCloudTransformation();

			//allows to maybe just compute transformed points on lower transformation levels
			void recompute_transforms(size_t until) const;

			[[nodiscard]] bool is_pointcloud_dirty() const;
			[[nodiscard]] bool is_transform_dirty() const;

			void set_transform_dirty(size_t i);

			std::vector<std::string> m_linknames;
			mutable std::vector<Matrix4f> m_transforms;
			std::unique_ptr<MetaPointCloud> m_links_meta_cloud;
			std::unique_ptr<MetaPointCloud> m_transformed_links_meta_cloud;

			
			mutable size_t dirty_pcl = 0;

			//indicates from where parts of the transformation or pcl have to be updated
			//0 means base_transformation also changed
			mutable size_t dirty_transforms = 0;

			/* host stored contents */
			//! pointer to the kinematic links
			std::map<std::string, KinematicLinkSharedPtr<convention>> m_links;
		};

	} // end of namespace
} // end of namespace
#endif