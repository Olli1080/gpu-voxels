// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/PrimitiveArray.h>

#include <cmath>
#include <string>

#include <gpu_voxels/vulkan/vulkan_common.h>

namespace gpu_voxels::vulkan
{
	namespace
	{
		bool is_finite(const PrimitiveEntry& p)
		{
			return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && std::isfinite(p.radius);
		}

		void check_size(float size, std::size_t index)
		{
			if (size < 0.0f)
				throw VulkanError("PrimitiveArray: the size of primitive " + std::to_string(index) + " is negative");
		}

		void check_diameter(float diameter)
		{
			if (diameter < 0.0f)
				throw VulkanError("PrimitiveArray: the diameter is negative");
		}
	}

	PrimitiveArray::PrimitiveArray(Dim3ui dimensions, float voxel_side_length, PrimitiveType type)
		: dimensions_(dimensions), voxel_side_length_(voxel_side_length), type_(type)
	{
		if (type != ePRIM_SPHERE && type != ePRIM_CUBOID)
			throw VulkanError("PrimitiveArray: unknown primitive type " + std::to_string(static_cast<unsigned>(type)));
		if (!(voxel_side_length > 0.0f) || !std::isfinite(voxel_side_length))
			throw VulkanError("PrimitiveArray: the voxel side length must be positive and finite");
	}

	void PrimitiveArray::store(std::vector<PrimitiveEntry>&& primitives)
	{
		for (std::size_t i = 0; i < primitives.size(); ++i)
			if (!is_finite(primitives[i]))
				throw VulkanError("PrimitiveArray: primitive " + std::to_string(i) + " is not finite in voxel units");
		primitives_ = std::move(primitives);
	}

	void PrimitiveArray::set_points(std::span<const Point4f> metric_points)
	{
		std::vector<PrimitiveEntry> converted;
		converted.reserve(metric_points.size());
		for (std::size_t i = 0; i < metric_points.size(); ++i)
		{
			const Point4f& p = metric_points[i];
			check_size(p.w, i);
			converted.push_back({ std::floor(p.x / voxel_side_length_) + 0.5f, std::floor(p.y / voxel_side_length_) + 0.5f,
				std::floor(p.z / voxel_side_length_) + 0.5f, (p.w / voxel_side_length_) / 2.f });
		}
		store(std::move(converted));
	}

	void PrimitiveArray::set_points(std::span<const Coord4i> voxel_points)
	{
		std::vector<PrimitiveEntry> converted;
		converted.reserve(voxel_points.size());
		for (std::size_t i = 0; i < voxel_points.size(); ++i)
		{
			const Coord4i& p = voxel_points[i];
			check_size(static_cast<float>(p.w), i);
			converted.push_back({ static_cast<float>(p.x) + 0.5f, static_cast<float>(p.y) + 0.5f, static_cast<float>(p.z) + 0.5f,
				static_cast<float>(p.w) / 2.f });
		}
		store(std::move(converted));
	}

	void PrimitiveArray::set_points(std::span<const Point3f> metric_points, float diameter)
	{
		check_diameter(diameter);
		std::vector<PrimitiveEntry> converted;
		converted.reserve(metric_points.size());
		for (const Point3f& p : metric_points)
			converted.push_back({ std::floor(p.x / voxel_side_length_) + 0.5f, std::floor(p.y / voxel_side_length_) + 0.5f,
				std::floor(p.z / voxel_side_length_) + 0.5f, (diameter / voxel_side_length_) / 2.f });
		store(std::move(converted));
	}

	void PrimitiveArray::set_points(std::span<const Coord3i> voxel_points, uint32_t diameter)
	{
		std::vector<PrimitiveEntry> converted;
		converted.reserve(voxel_points.size());
		for (const Coord3i& p : voxel_points)
			converted.push_back({ static_cast<float>(p.x) + 0.5f, static_cast<float>(p.y) + 0.5f, static_cast<float>(p.z) + 0.5f,
				static_cast<float>(diameter) / 2.f });
		store(std::move(converted));
	}
}
