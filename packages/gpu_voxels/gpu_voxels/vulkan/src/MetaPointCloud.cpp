// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/MetaPointCloud.h>

#include <algorithm>
#include <cstring>
#include <numeric>

#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>

#include "Shaders.h"

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kLocalSize = 64;
		constexpr vk::DeviceSize kMinBufferBytes = sizeof(Point3f);

		struct TransformPush { uint32_t point_count, entry_count; };

		/// Table entry of the transform shaders: begin, end, enabled, pad, 3 matrix rows (row-major).
		struct TableEntry
		{
			uint32_t begin, end, enabled, pad;
			float m[12];
		};
		static_assert(sizeof(TableEntry) == 64);

		uint32_t group_count(const Context& context, uint64_t items)
		{
			const uint64_t needed = (items + kLocalSize - 1) / kLocalSize;
			const uint64_t cap = std::min<uint64_t>(context.limits().maxComputeWorkGroupCount[0], 65535);
			return static_cast<uint32_t>(std::clamp<uint64_t>(needed, 1, cap));
		}

		std::vector<uint32_t> offsets_from_sizes(std::span<const uint32_t> sizes)
		{
			std::vector<uint32_t> offsets(sizes.size() + 1, 0);
			uint64_t total = 0;
			for (std::size_t i = 0; i < sizes.size(); ++i)
			{
				total += sizes[i];
				if (total > UINT32_MAX / 3)
					throw VulkanError("MetaPointCloud: too many points");
				offsets[i + 1] = static_cast<uint32_t>(total);
			}
			return offsets;
		}
	}

	struct MetaPointCloud::Kernels
	{
		explicit Kernels(Context& context)
			: transform(context, GVL_SHADER_SPAN(meta_transform_comp), 3, sizeof(TransformPush)),
			  transform_inplace(context, GVL_SHADER_SPAN(meta_transform_inplace_comp), 2, sizeof(TransformPush))
		{
		}

		ComputePipeline transform;
		ComputePipeline transform_inplace;
	};

	MetaPointCloud::MetaPointCloud(Context& context, std::span<const uint32_t> cloud_sizes)
		: context_(&context), offsets_(offsets_from_sizes(cloud_sizes)),
		  points_(std::make_unique<DeviceBuffer>(context, std::max<vk::DeviceSize>(kMinBufferBytes,
			  static_cast<vk::DeviceSize>(offsets_.back()) * sizeof(Point3f)))),
		  kernels_(std::make_shared<Kernels>(context))
	{
		points_->fill(0);
	}

	MetaPointCloud::MetaPointCloud(Context& context, std::span<const std::vector<Point3f>> clouds)
		: context_(&context), points_(nullptr), kernels_(std::make_shared<Kernels>(context))
	{
		std::vector<uint32_t> sizes;
		for (const auto& c : clouds)
			sizes.push_back(static_cast<uint32_t>(c.size()));
		offsets_ = offsets_from_sizes(sizes);
		points_ = std::make_unique<DeviceBuffer>(context, std::max<vk::DeviceSize>(kMinBufferBytes,
			static_cast<vk::DeviceSize>(offsets_.back()) * sizeof(Point3f)));
		points_->fill(0);

		std::vector<Point3f> all;
		all.reserve(offsets_.back());
		for (const auto& c : clouds)
			all.insert(all.end(), c.begin(), c.end());
		if (!all.empty())
			points_->upload(all.data(), all.size() * sizeof(Point3f));
	}

	MetaPointCloud MetaPointCloud::make_like() const
	{
		std::vector<uint32_t> sizes(num_clouds());
		for (uint32_t i = 0; i < sizes.size(); ++i)
			sizes[i] = cloud_size(i);
		MetaPointCloud other(*context_, sizes);
		other.kernels_ = kernels_;
		return other;
	}

	void MetaPointCloud::check_cloud(uint32_t cloud) const
	{
		if (cloud >= num_clouds())
			throw VulkanError("MetaPointCloud: sub cloud " + std::to_string(cloud) + " does not exist");
	}

	void MetaPointCloud::check_compatible(const MetaPointCloud& out) const
	{
		if (out.context_ != context_)
			throw VulkanError("MetaPointCloud: different contexts");
		if (out.offsets_ != offsets_)
			throw VulkanError("MetaPointCloud: target has different sub cloud sizes");
	}

	void MetaPointCloud::update_cloud(uint32_t cloud, std::span<const Point3f> points)
	{
		check_cloud(cloud);
		if (points.size() != cloud_size(cloud))
			throw VulkanError("MetaPointCloud::update_cloud: the number of points must not change");
		if (!points.empty())
			points_->upload(points.data(), points.size_bytes(), static_cast<vk::DeviceSize>(offsets_[cloud]) * sizeof(Point3f));
	}

	std::vector<Point3f> MetaPointCloud::download() const
	{
		std::vector<Point3f> out(accumulated_size());
		if (!out.empty())
			points_->download(out.data(), out.size() * sizeof(Point3f));
		return out;
	}

	std::vector<Point3f> MetaPointCloud::download_cloud(uint32_t cloud) const
	{
		check_cloud(cloud);
		std::vector<Point3f> out(cloud_size(cloud));
		if (!out.empty())
			points_->download(out.data(), out.size() * sizeof(Point3f), static_cast<vk::DeviceSize>(offsets_[cloud]) * sizeof(Point3f));
		return out;
	}

	void MetaPointCloud::transform(const Matrix4f& transformation, MetaPointCloud& out) const
	{
		std::vector<SubCloudTransform> all;
		for (uint32_t i = 0; i < num_clouds(); ++i)
			all.push_back({ static_cast<uint16_t>(i), transformation });
		transform_sub_clouds(all, out);
	}

	void MetaPointCloud::transform_sub_cloud(uint16_t cloud, const Matrix4f& transformation, MetaPointCloud& out) const
	{
		const SubCloudTransform single{ cloud, transformation };
		transform_sub_clouds(std::span<const SubCloudTransform>(&single, 1), out);
	}

	namespace
	{
		std::vector<TableEntry> build_table(const std::vector<uint32_t>& offsets, std::span<const SubCloudTransform> transformations)
		{
			const uint32_t clouds = static_cast<uint32_t>(offsets.size()) - 1;
			std::vector<TableEntry> table(clouds);
			for (uint32_t i = 0; i < clouds; ++i)
				table[i] = TableEntry{ offsets[i], offsets[i + 1], 0u, 0u, {} };
			for (const auto& t : transformations)
			{
				if (t.cloud >= clouds)
					throw VulkanError("MetaPointCloud: sub cloud " + std::to_string(t.cloud) + " does not exist");
				auto& e = table[t.cloud];
				e.enabled = 1;
				for (int row = 0; row < 3; ++row)
					for (int col = 0; col < 4; ++col)
						e.m[row * 4 + col] = t.transform(row, col);
			}
			return table;
		}
	}

	void MetaPointCloud::transform_sub_clouds(std::span<const SubCloudTransform> transformations, MetaPointCloud& out) const
	{
		check_compatible(out);
		if (empty() || transformations.empty())
			return;
		if (&out == this)
			throw VulkanError("MetaPointCloud: use transform_self for in-place transformations");

		const auto table = build_table(offsets_, transformations);
		DeviceBuffer table_buffer(*context_, table.size() * sizeof(TableEntry));
		table_buffer.upload(table.data(), table.size() * sizeof(TableEntry));

		const TransformPush push{ accumulated_size(), static_cast<uint32_t>(table.size()) };
		CommandBatch batch(*context_);
		batch.dispatch(kernels_->transform, { points_.get(), out.points_.get(), &table_buffer },
			group_count(*context_, accumulated_size()), push);
		batch.submit_and_wait();
	}

	void MetaPointCloud::transform_self(const Matrix4f& transformation)
	{
		std::vector<SubCloudTransform> all;
		for (uint32_t i = 0; i < num_clouds(); ++i)
			all.push_back({ static_cast<uint16_t>(i), transformation });
		if (empty())
			return;
		const auto table = build_table(offsets_, all);
		DeviceBuffer table_buffer(*context_, table.size() * sizeof(TableEntry));
		table_buffer.upload(table.data(), table.size() * sizeof(TableEntry));
		const TransformPush push{ accumulated_size(), static_cast<uint32_t>(table.size()) };
		CommandBatch batch(*context_);
		batch.dispatch(kernels_->transform_inplace, { points_.get(), &table_buffer }, group_count(*context_, accumulated_size()), push);
		batch.submit_and_wait();
	}

	void MetaPointCloud::transform_self_sub_cloud(uint16_t cloud, const Matrix4f& transformation)
	{
		check_cloud(cloud);
		if (empty())
			return;
		const SubCloudTransform single{ cloud, transformation };
		const auto table = build_table(offsets_, std::span<const SubCloudTransform>(&single, 1));
		DeviceBuffer table_buffer(*context_, table.size() * sizeof(TableEntry));
		table_buffer.upload(table.data(), table.size() * sizeof(TableEntry));
		const TransformPush push{ accumulated_size(), static_cast<uint32_t>(table.size()) };
		CommandBatch batch(*context_);
		batch.dispatch(kernels_->transform_inplace, { points_.get(), &table_buffer }, group_count(*context_, accumulated_size()), push);
		batch.submit_and_wait();
	}
}
