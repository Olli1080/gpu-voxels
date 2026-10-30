// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Serial robot described by Denavit-Hartenberg parameters whose link point clouds are moved on the GPU.
 *
 * Counterpart of robot::KinematicChain / KinematicLink (CLASSIC and CRAIGS convention). Transform of link i:
 *   T_0 = base,  T_{i+1} = T_i * DH_i(joint value i)      (T_0 is the base, T_n the end effector)
 * and the point cloud of link i is transformed with T_i (as in CUDA: link 0 sits at the base transformation).
 * Joint dependent transformations are computed on the CPU in float (as in CUDA), the point clouds on the GPU.
 */
#ifndef GPU_VOXELS_VULKAN_KINEMATIC_CHAIN_H_INCLUDED
#define GPU_VOXELS_VULKAN_KINEMATIC_CHAIN_H_INCLUDED

#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	enum class DHConvention { Classic, Craigs };
	enum class DHJointType { Revolute, Prismatic };

	struct DHParameters
	{
		float d = 0, theta = 0, a = 0, alpha = 0; ///< constant DH parameters
		float value = 0;                          ///< joint value, added to theta (revolute) or d (prismatic)
		DHJointType joint_type = DHJointType::Revolute;
	};

	/// Homogeneous matrix of one DH link (CUDA DHParameters<convention>::convertDHtoM).
	[[nodiscard]] Matrix4f dh_matrix(DHConvention convention, const DHParameters& parameters);

	struct LinkSpec
	{
		std::string name;
		DHParameters dh;
		std::optional<std::vector<Point3f>> cloud; ///< links without cloud only contribute their DH transformation
	};

	class KinematicChain
	{
	public:
		/// Sub cloud k of the transformed clouds belongs to the k-th link that has a cloud.
		KinematicChain(Context& context, DHConvention convention, std::span<const LinkSpec> links,
			const Matrix4f& base_transformation = Matrix4f::identity());

		[[nodiscard]] std::size_t num_links() const { return links_.size(); }
		[[nodiscard]] const std::vector<std::string>& link_names() const { return names_; }
		/// Index of the sub cloud of a link, empty if the link has no cloud.
		[[nodiscard]] std::optional<uint32_t> cloud_index(const std::string& link_name) const;

		/// Joint values by link name; unknown names are ignored (like CUDA).
		void set_configuration(const std::map<std::string, float>& joint_values);
		[[nodiscard]] std::map<std::string, float> configuration() const;
		void set_base_transformation(const Matrix4f& base);
		[[nodiscard]] const Matrix4f& base_transformation() const { return transforms_.front(); }

		/// transform(0) is the base, transform(i) the frame of link i, transform(num_links()) the end effector.
		/// \throws VulkanError for idx > num_links()
		[[nodiscard]] Matrix4f transform(std::size_t idx) const;

		/// Replace the (untransformed) cloud of a link; the number of points must not change.
		void update_link_cloud(const std::string& link_name, std::span<const Point3f> points);

		/// Untransformed link clouds.
		[[nodiscard]] const MetaPointCloud& link_clouds() const { return link_clouds_; }
		/// Link clouds in world coordinates for the current configuration (computed on the GPU when needed).
		[[nodiscard]] const MetaPointCloud& transformed_clouds();

	private:
		void recompute_transforms() const;

		DHConvention convention_;
		std::vector<std::string> names_;
		std::vector<DHParameters> links_;
		std::vector<std::optional<uint32_t>> cloud_of_link_;
		mutable std::vector<Matrix4f> transforms_; ///< size num_links + 1
		mutable bool transforms_dirty_ = true;
		MetaPointCloud link_clouds_;
		MetaPointCloud transformed_;
		bool clouds_dirty_ = true;
	};
}

#endif
