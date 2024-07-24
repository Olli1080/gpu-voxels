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
			const Matrix4f& base_transformation)
		{
			m_links_meta_cloud = std::make_unique<MetaPointCloud>(pointclouds);
			init(linknames, dh_params, base_transformation);
		}

		template<DHConvention convention>
		KinematicChain<convention>::KinematicChain(const std::vector<std::string>& linknames,
			const std::vector<robot::DHParameters<convention>>& dh_params,
			const std::vector<std::string>& paths_to_pointclouds,
			bool use_model_path,
			const Matrix4f& base_transformation)
		{
			// The untransformed point clouds
			m_links_meta_cloud = std::make_unique<MetaPointCloud>(paths_to_pointclouds, paths_to_pointclouds, use_model_path);
			init(linknames, dh_params, base_transformation);
		}

		template<DHConvention convention>
		void KinematicChain<convention>::init(const std::vector<std::string>& linknames,
			const std::vector<robot::DHParameters<convention>>& dh_params,
			const Matrix4f& base_transformation)
		{
			// sanity check:
			if (linknames.size() != dh_params.size())
			{
				LOGGING_ERROR_C(RobotLog, KinematicChain,
					"Number of linknames does not fit number of DH parameters. EXITING!" << endl);
				exit(-1);
			}
			const size_t size = linknames.size();
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

			m_linknames = linknames;
			m_transforms = std::vector<Matrix4f>(size + 1, Matrix4f::Identity());
			m_transforms.front() = base_transformation;

			for (size_t i = 0; i < size; ++i)
				m_links[linknames[i]] = std::make_shared<KinematicLink<convention>>(dh_params[i]);


			LOGGING_INFO_C(RobotLog, KinematicChain, "now handling " << m_links.size() << " links." << endl);

			// allocate a copy of the pointclouds to store the transformed clouds (host and device)
			m_transformed_links_meta_cloud = std::make_unique<MetaPointCloud>(*m_links_meta_cloud);

		}

		template<DHConvention convention>
		void KinematicChain<convention>::performPointCloudTransformation()
		{
			if (!is_pointcloud_dirty())
				return;

			recompute_transforms(m_transforms.size());

			std::list<std::tuple<uint16_t, Matrix4f>> subcloud_transforms;
			// Iterate over all links and transform pointclouds with the according name
			// if no pointcloud was found, still the transformation has to be calculated for the next link.
			size_t idx = 0;
			for (const auto& linkname : m_linknames)
			{
				if (idx >= dirty_pcl)
				{
					const auto pc_num = m_links_meta_cloud->getCloudNumber(linkname);
					if (pc_num.has_value())
						subcloud_transforms.emplace_back(pc_num.value(), m_transforms[idx]);
				}
				++idx;
			}

			m_links_meta_cloud->transformSubClouds(subcloud_transforms, *m_transformed_links_meta_cloud);

			dirty_pcl = m_linknames.size();
		}

		template<DHConvention convention>
		void KinematicChain<convention>::recompute_transforms(size_t until) const
		{
			if (until == 0) //because of unsigned loop comparison
				return;

			for (size_t i = dirty_transforms; i < until - 1; ++i)
			{
				Matrix4f m_dh_transformation;
				m_links.at(m_linknames[i])->getMatrixRepresentation(m_dh_transformation);
				m_transforms[i + 1] = m_transforms[i] * m_dh_transformation;
			}
			dirty_transforms = m_transforms.size();
		}

		template<DHConvention convention>
		bool KinematicChain<convention>::is_pointcloud_dirty() const
		{
			return dirty_pcl < m_linknames.size();
		}

		template<DHConvention convention>
		bool KinematicChain<convention>::is_transform_dirty() const
		{
			return dirty_transforms < m_transforms.size();
		}

		template<DHConvention convention>
		void KinematicChain<convention>::set_transform_dirty(size_t i)
		{
			if (i >= dirty_transforms)
				return;

			dirty_transforms = i;
			dirty_pcl = (std::min)(i, dirty_pcl);
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

				if (dirty_transforms > 0)
					set_transform_dirty(std::distance(m_linknames.begin(), std::ranges::find(m_linknames, name)));
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
			//TODO:: potentially only this pointcloud has to be transformed but it does also transform all following ones
			dirty_pcl = std::distance(m_linknames.begin(), std::ranges::find(m_linknames, link_name));
			m_links_meta_cloud->updatePointCloud(link_name, cloud, true);
		}

		template<DHConvention convention>
		void KinematicChain<convention>::setBaseTransformation(const Matrix4f& base_transformation)
		{
			if (base_transformation == m_transforms.front())
				return;

			set_transform_dirty(0);
			m_transforms.front() = base_transformation;
		}

		template<DHConvention convention>
		void KinematicChain<convention>::getBaseTransformation(Matrix4f& base_transformation) const
		{
			base_transformation = m_transforms.front();
		}
		template<DHConvention convention>
		Matrix4f KinematicChain<convention>::getTransform(size_t idx) const
		{
			if (idx >= m_transforms.size())
			{
				LOGGING_ERROR_C(RobotLog, KinematicChain,
					"Invalid idx for transform! returning identity instead" << endl);
				return Matrix4f::Identity();
			}

			recompute_transforms(idx + 1);

			return m_transforms[idx];
		}

		/*
		template<DHConvention convention>
		Vector3f KinematicChain<convention>::transform_point(const Vector3f& p) const
		{
			recompute_transforms(m_transforms.size());

			const auto& trafo = m_transforms.back();

			Vector3f result = trafo.block<3, 1>(0, 3);
			result += trafo.block<3, 3>(0, 0) * p;

			return result;
		}*/

		template class KinematicChain<CLASSIC>;
		template class KinematicChain<CRAIGS>;

	} // end of namespace
} // end of namespace