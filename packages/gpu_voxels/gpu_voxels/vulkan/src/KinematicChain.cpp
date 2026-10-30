// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/KinematicChain.h>

#include <cmath>

namespace gpu_voxels::vulkan
{
	namespace
	{
		std::vector<std::vector<Point3f>> collect_clouds(std::span<const LinkSpec> links)
		{
			std::vector<std::vector<Point3f>> clouds;
			for (const auto& link : links)
				if (link.cloud)
					clouds.push_back(*link.cloud);
			return clouds;
		}
	}

	Matrix4f dh_matrix(DHConvention convention, const DHParameters& p)
	{
		float d = p.d;
		float theta = p.theta;
		switch (p.joint_type)
		{
		case DHJointType::Prismatic: d = p.d + p.value; break;
		case DHJointType::Revolute: theta = p.theta + p.value; break;
		default: throw VulkanError("dh_matrix: illegal joint type");
		}

		const float ca = std::cos(p.alpha);
		const float sa = std::sin(p.alpha);
		const float ct = std::cos(theta);
		const float st = std::sin(theta);

		Matrix4f m;
		if (convention == DHConvention::Classic)
		{
			m(0, 0) = ct;  m(0, 1) = -st * ca; m(0, 2) = st * sa;  m(0, 3) = p.a * ct;
			m(1, 0) = st;  m(1, 1) = ct * ca;  m(1, 2) = -ct * sa; m(1, 3) = p.a * st;
			m(2, 0) = 0.f; m(2, 1) = sa;       m(2, 2) = ca;       m(2, 3) = d;
		}
		else // Craig
		{
			m(0, 0) = ct;      m(0, 1) = -st;     m(0, 2) = 0.f; m(0, 3) = p.a;
			m(1, 0) = st * ca; m(1, 1) = ct * ca; m(1, 2) = -sa; m(1, 3) = -d * sa;
			m(2, 0) = st * sa; m(2, 1) = ct * sa; m(2, 2) = ca;  m(2, 3) = d * ca;
		}
		m(3, 0) = 0.f; m(3, 1) = 0.f; m(3, 2) = 0.f; m(3, 3) = 1.f;
		return m;
	}

	KinematicChain::KinematicChain(Context& context, DHConvention convention, std::span<const LinkSpec> links,
		const Matrix4f& base_transformation)
		: convention_(convention), transforms_(links.size() + 1, Matrix4f::identity()),
		  link_clouds_(context, collect_clouds(links)), transformed_(link_clouds_.make_like())
	{
		transforms_.front() = base_transformation;
		uint32_t next_cloud = 0;
		for (const auto& link : links)
		{
			names_.push_back(link.name);
			links_.push_back(link.dh);
			cloud_of_link_.push_back(link.cloud ? std::optional<uint32_t>(next_cloud++) : std::nullopt);
		}
		for (std::size_t i = 0; i < names_.size(); ++i)
			for (std::size_t j = i + 1; j < names_.size(); ++j)
				if (names_[i] == names_[j])
					throw VulkanError("KinematicChain: duplicate link name '" + names_[i] + "'");
	}

	std::optional<uint32_t> KinematicChain::cloud_index(const std::string& link_name) const
	{
		for (std::size_t i = 0; i < names_.size(); ++i)
			if (names_[i] == link_name)
				return cloud_of_link_[i];
		return std::nullopt;
	}

	void KinematicChain::set_configuration(const std::map<std::string, float>& joint_values)
	{
		for (std::size_t i = 0; i < names_.size(); ++i)
		{
			const auto it = joint_values.find(names_[i]);
			if (it == joint_values.end() || links_[i].value == it->second)
				continue;
			links_[i].value = it->second;
			transforms_dirty_ = true;
			clouds_dirty_ = true;
		}
	}

	std::map<std::string, float> KinematicChain::configuration() const
	{
		std::map<std::string, float> result;
		for (std::size_t i = 0; i < names_.size(); ++i)
			result[names_[i]] = links_[i].value;
		return result;
	}

	void KinematicChain::set_base_transformation(const Matrix4f& base)
	{
		if (base == transforms_.front())
			return;
		transforms_.front() = base;
		transforms_dirty_ = true;
		clouds_dirty_ = true;
	}

	void KinematicChain::recompute_transforms() const
	{
		if (!transforms_dirty_)
			return;
		for (std::size_t i = 0; i < links_.size(); ++i)
			transforms_[i + 1] = transforms_[i] * dh_matrix(convention_, links_[i]);
		transforms_dirty_ = false;
	}

	Matrix4f KinematicChain::transform(std::size_t idx) const
	{
		if (idx > links_.size())
			throw VulkanError("KinematicChain::transform: index out of range");
		recompute_transforms();
		return transforms_[idx];
	}

	void KinematicChain::update_link_cloud(const std::string& link_name, std::span<const Point3f> points)
	{
		const auto cloud = cloud_index(link_name);
		if (!cloud)
			throw VulkanError("KinematicChain::update_link_cloud: link '" + link_name + "' has no cloud");
		link_clouds_.update_cloud(*cloud, points);
		clouds_dirty_ = true;
	}

	const MetaPointCloud& KinematicChain::transformed_clouds()
	{
		if (clouds_dirty_)
		{
			recompute_transforms();
			std::vector<SubCloudTransform> transformations;
			for (std::size_t i = 0; i < links_.size(); ++i)
				if (cloud_of_link_[i])
					transformations.push_back({ static_cast<uint16_t>(*cloud_of_link_[i]), transforms_[i] });
			link_clouds_.transform_sub_clouds(transformations, transformed_);
			clouds_dirty_ = false;
		}
		return transformed_;
	}
}
