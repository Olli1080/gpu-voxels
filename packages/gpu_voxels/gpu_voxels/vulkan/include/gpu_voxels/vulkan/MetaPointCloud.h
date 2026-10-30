// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Several point clouds ("sub clouds", e.g. robot links) in one device buffer, with GPU transformation.
 *
 * Counterpart of the CUDA MetaPointCloud: the accumulated points live in one buffer, sub cloud i occupies
 * [offset(i), offset(i) + size(i)). Points are stored as packed floats (x, y, z), stride 12 bytes.
 * Differences to CUDA: no host mirror (download on demand); size mismatches between source and target of a
 * transformation throw VulkanError instead of logging an error and silently doing nothing.
 */
#ifndef GPU_VOXELS_VULKAN_META_POINT_CLOUD_H_INCLUDED
#define GPU_VOXELS_VULKAN_META_POINT_CLOUD_H_INCLUDED

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	/// Transformation of one sub cloud.
	struct SubCloudTransform
	{
		uint16_t cloud;
		Matrix4f transform;
	};

	class MetaPointCloud
	{
	public:
		/// One sub cloud per entry of `clouds` (empty sub clouds are allowed).
		MetaPointCloud(Context& context, std::span<const std::vector<Point3f>> clouds);
		/// Sub clouds of the given sizes, all points initialised to (0, 0, 0).
		MetaPointCloud(Context& context, std::span<const uint32_t> cloud_sizes);

		MetaPointCloud(MetaPointCloud&&) noexcept = default;
		MetaPointCloud& operator=(MetaPointCloud&&) noexcept = default;
		MetaPointCloud(const MetaPointCloud&) = delete;
		MetaPointCloud& operator=(const MetaPointCloud&) = delete;

		/// A cloud with the same sub cloud sizes and zero points (target for transformations).
		[[nodiscard]] MetaPointCloud make_like() const;

		[[nodiscard]] uint32_t num_clouds() const { return static_cast<uint32_t>(offsets_.size()) - 1; }
		[[nodiscard]] uint32_t cloud_size(uint32_t cloud) const { return offsets_.at(cloud + 1) - offsets_.at(cloud); }
		[[nodiscard]] uint32_t cloud_offset(uint32_t cloud) const { return offsets_.at(cloud); }
		[[nodiscard]] uint32_t accumulated_size() const { return offsets_.back(); }
		[[nodiscard]] bool empty() const { return accumulated_size() == 0; }
		/// num_clouds() + 1 ascending offsets; the last entry is the accumulated size.
		[[nodiscard]] const std::vector<uint32_t>& offsets() const { return offsets_; }
		/// Device buffer with the packed points (at least 12 bytes even if the cloud is empty).
		[[nodiscard]] const DeviceBuffer& points() const { return *points_; }
		[[nodiscard]] DeviceBuffer& points() { return *points_; }
		[[nodiscard]] Context& context() const { return *context_; }

		/// Replace the points of one sub cloud (the size must not change).
		void update_cloud(uint32_t cloud, std::span<const Point3f> points);
		[[nodiscard]] std::vector<Point3f> download() const;
		[[nodiscard]] std::vector<Point3f> download_cloud(uint32_t cloud) const;

		/// All sub clouds, p' = R * p + t. `out` must have identical sub cloud sizes.
		void transform(const Matrix4f& transformation, MetaPointCloud& out) const;
		void transform_sub_cloud(uint16_t cloud, const Matrix4f& transformation, MetaPointCloud& out) const;
		/// Several sub clouds with individual transformations in one GPU dispatch; other sub clouds of `out` are untouched.
		void transform_sub_clouds(std::span<const SubCloudTransform> transformations, MetaPointCloud& out) const;
		void transform_self(const Matrix4f& transformation);
		void transform_self_sub_cloud(uint16_t cloud, const Matrix4f& transformation);

		/// Pipelines shared by all MetaPointClouds of a context (internal).
		struct Kernels;

	private:
		void check_compatible(const MetaPointCloud& out) const;
		void check_cloud(uint32_t cloud) const;

		Context* context_;
		std::vector<uint32_t> offsets_;
		std::unique_ptr<DeviceBuffer> points_;
		std::shared_ptr<Kernels> kernels_;
	};
}

#endif
