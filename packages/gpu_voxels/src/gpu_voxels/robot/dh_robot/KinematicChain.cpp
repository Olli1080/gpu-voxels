// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
 //----------------------------------------------------------------------
#include "KinematicChain.h"

#include <string>
#include <map>
#include <ranges>

#include <gpu_voxels/logging/logging_robot.h>

namespace gpu_voxels
{
	namespace robot
	{
		template<DHConvention convention>
		KinematicChain<convention>::KinematicChain(const std::vector<std::string>& linknames,
			const std::vector<robot::DHParameters<convention>>& dh_params,
			const MetaPointCloud& pointclouds,
			Matrix4f base_transformation)
			: m_base_transformation(std::move(base_transformation))
		{
			m_links_meta_cloud = std::make_unique<MetaPointCloud>(pointclouds);
			init(linknames, dh_params);
		}

		template<DHConvention convention>
		KinematicChain<convention>::KinematicChain(const std::vector<std::string>& linknames,
			const std::vector<robot::DHParameters<convention>>& dh_params,
			const std::vector<std::string>& paths_to_pointclouds,
			bool use_model_path,
			Matrix4f base_transformation)
			: m_base_transformation(std::move(base_transformation))
		{
			// The untransformed point clouds
			m_links_meta_cloud = std::make_unique<MetaPointCloud>(paths_to_pointclouds, paths_to_pointclouds, use_model_path);
			init(linknames, dh_params);
		}

		template<DHConvention convention>
		void KinematicChain<convention>::init(const std::vector<std::string>& linknames,
			const std::vector<robot::DHParameters<convention>>& dh_params)
		{
			m_linknames = linknames;

			// sanity check:
			if (linknames.size() != dh_params.size())
			{
				LOGGING_ERROR_C(RobotLog, KinematicChain,
					"Number of linknames does not fit number of DH parameters. EXITING!" << endl);
				exit(-1);
			}
			//  std::map<uint16_t, std::string> cloud_names = m_links_meta_cloud->getCloudNames();
			//  for (size_t i = 0; i < linknames.size(); i++)
			//  {
			//    if(cloud_names[i] != linknames[i])
			//    {
			//      LOGGING_ERROR_C(RobotLog, KinematicChain,
			//                      "Names of clouds differ from names of links. EXITING!" << endl);
			//      exit(-1);
			//    }
			//  }

			for (size_t i = 0; i < dh_params.size(); i++)
				m_links[linknames[i]] = std::make_shared<KinematicLink<convention>>(dh_params[i]);


			LOGGING_INFO_C(RobotLog, KinematicChain, "now handling " << m_links.size() << " links." << endl);

			// allocate a copy of the pointclouds to store the transformed clouds (host and device)
			m_transformed_links_meta_cloud = std::make_unique<MetaPointCloud>(*m_links_meta_cloud);

		}

		template<DHConvention convention>
		void KinematicChain<convention>::performPointCloudTransformation()
		{
			if (dirty >= static_cast<int>(m_linknames.size()))
				return;

			Matrix4f transformation = m_base_transformation;

			std::list<std::tuple<uint16_t, Matrix4f>> subcloud_transforms;
			// Iterate over all links and transform pointclouds with the according name
			// if no pointcloud was found, still the transformation has to be calculated for the next link.
			int idx = 0;
			for (const auto& linkname : m_linknames)
			{
				if (idx >= dirty)
				{
					const auto pc_num = m_links_meta_cloud->getCloudNumber(linkname);
					if (pc_num.has_value())
					{
						subcloud_transforms.emplace_back(pc_num.value(), transformation);
						//m_links_meta_cloud->transformSubCloud(pc_num.value(), transformation, *m_transformed_links_meta_cloud);
					}
				}
				// TODO:: this documentation is outdated
				// Sending the actual transformation for this link to the GPU.
				// This means the DH Transformation i is not applied to link-pointcloud i,
				// but to link pointcloud i+1, i+2...
				Matrix4f m_dh_transformation;
				m_links[linkname]->getMatrixRepresentation(m_dh_transformation);
				transformation = transformation * m_dh_transformation;
				//std::cout << "Trafo Matrix ["<< linkname <<"] = " << m_dh_transformation  << std::endl;
				//std::cout << "Accumulated Trafo Matrix ["<< linkname <<"] = " << transformation << std::endl;
				++idx;
			}

			m_links_meta_cloud->transformSubClouds(subcloud_transforms, *m_transformed_links_meta_cloud);

			dirty = m_linknames.size();
		}

		template<DHConvention convention>
		void KinematicChain<convention>::setConfiguration(const JointValueMap& joint_values)
		{
			for (auto& [name, link] : m_links)
			{
				if (!joint_values.contains(name))
					continue;
				
				float new_joint_value = joint_values.at(name);
				if (link->getJointValue() == new_joint_value)
					continue;

				link->setJointValue(joint_values.at(name));

				if (dirty > 0)
				{
					auto pos = std::ranges::find(m_linknames, name);
					dirty = (std::min)(static_cast<int>(std::distance(m_linknames.begin(), pos)), dirty);
				}
			}
		}

		template<DHConvention convention>
		void KinematicChain<convention>::getJointNames(std::vector<std::string>& jointnames)
		{
			jointnames.clear();
			jointnames.reserve(m_links.size());
			for (auto name : m_links | std::views::keys)
				jointnames.emplace_back(name);
		}

		template<DHConvention convention>
		void KinematicChain<convention>::getConfiguration(JointValueMap& joint_values)
		{
			for (auto [name, link] : m_links)
				joint_values[name] = link->getJointValue();
		}

		template<DHConvention convention>
		void KinematicChain<convention>::getLowerJointLimits(JointValueMap& lower_limits)
		{
			LOGGING_ERROR_C(RobotLog, KinematicChain,
				"getLowerJointLimits not implemented for DH-Robot!" << endl);
		}

		template<DHConvention convention>
		void KinematicChain<convention>::getUpperJointLimits(JointValueMap& upper_limits)
		{
			LOGGING_ERROR_C(RobotLog, KinematicChain,
				"getUpperJointLimits not implemented for DH-Robot!" << endl);
		}

		template<DHConvention convention>
		const MetaPointCloud* KinematicChain<convention>::getTransformedClouds()
		{
			performPointCloudTransformation();
			return m_transformed_links_meta_cloud.get();
		}

		template<DHConvention convention>
		void KinematicChain<convention>::updatePointcloud(const std::string& link_name, const std::vector<Vector3f>& cloud)
		{
			dirty = -1;
			m_links_meta_cloud->updatePointCloud(link_name, cloud, true);
		}

		template<DHConvention convention>
		void KinematicChain<convention>::setBaseTransformation(const Matrix4f& base_transformation)
		{
			dirty = -1;
			m_base_transformation = base_transformation;
		}

		template<DHConvention convention>
		void KinematicChain<convention>::getBaseTransformation(Matrix4f& base_transformation) const
		{
			base_transformation = m_base_transformation;
		}

		template class KinematicChain<CLASSIC>;
		template class KinematicChain<CRAIGS>;

	} // end of namespace
} // end of namespace