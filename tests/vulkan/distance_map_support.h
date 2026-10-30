// Helpers shared by the distance map tests (slice S5).
#ifndef GVL_TEST_DISTANCE_MAP_SUPPORT_H_INCLUDED
#define GVL_TEST_DISTANCE_MAP_SUPPORT_H_INCLUDED

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <gpu_voxels/vulkan/DistanceVoxelMap.h>

#include "../reference/distance_map_reference.h"

namespace gvl_test::distance
{
	using namespace gpu_voxels::vulkan;
	using gvl_reference::DistanceMapReference;

	inline constexpr float kUnitSide = 1.0f;

	inline DistanceMapReference distance_reference(Dim3ui d, float side = kUnitSide) { return DistanceMapReference(d.x, d.y, d.z, side); }

	inline std::vector<Coord3ui> random_coordinates(std::mt19937& rng, std::size_t n, Dim3ui d)
	{
		std::uniform_int_distribution<uint32_t> vx(0, d.x - 1), vy(0, d.y - 1), vz(0, d.z - 1);
		std::vector<Coord3ui> out(n);
		for (auto& c : out)
			c = { vx(rng), vy(rng), vz(rng) };
		return out;
	}

	/// Brute force squared distance of every voxel (-1 without obstacles), in linear index order.
	inline std::vector<int64_t> expected_squared_distances(const DistanceMapReference& ref, Dim3ui d)
	{
		std::vector<int64_t> out;
		out.reserve(static_cast<std::size_t>(d.voxel_count()));
		for (uint32_t z = 0; z < d.z; ++z)
			for (uint32_t y = 0; y < d.y; ++y)
				for (uint32_t x = 0; x < d.x; ++x)
					out.push_back(ref.min_squared_distance(x, y, z));
		return out;
	}

	/// Every voxel must store an obstacle at the minimal distance (`expected` from expected_squared_distances()), or be
	/// uninitialised without obstacles.
	inline ::testing::AssertionResult is_exact_distance_field(const DistanceVoxelMap& map, const DistanceMapReference& ref,
		const std::vector<int64_t>& expected_distances)
	{
		const Dim3ui d = map.dimensions();
		const auto sites = map.download_obstacles();
		for (uint32_t z = 0; z < d.z; ++z)
			for (uint32_t y = 0; y < d.y; ++y)
				for (uint32_t x = 0; x < d.x; ++x)
				{
					const uint64_t index = DistanceVoxelMap::voxel_index(d, { x, y, z });
					const Coord3ui site = sites[index];
					const int64_t expected = expected_distances[index];
					if (expected < 0)
					{
						if (site.x != 1023 || site.y != 1023 || site.z != 1023)
							return ::testing::AssertionFailure() << "voxel (" << x << "," << y << "," << z << ") has an obstacle but there is none";
						continue;
					}
					if (!ref.is_obstacle(site.x, site.y, site.z))
						return ::testing::AssertionFailure() << "voxel (" << x << "," << y << "," << z << ") stores (" << site.x << "," << site.y
							<< "," << site.z << "), which is no obstacle";
					const int64_t dx = static_cast<int64_t>(x) - site.x, dy = static_cast<int64_t>(y) - site.y, dz = static_cast<int64_t>(z) - site.z;
					if (dx * dx + dy * dy + dz * dz != expected)
						return ::testing::AssertionFailure() << "voxel (" << x << "," << y << "," << z << ") squared distance " << dx * dx + dy * dy + dz * dz
							<< " expected " << expected;
				}
		return ::testing::AssertionSuccess();
	}

	inline ::testing::AssertionResult is_exact_distance_field(const DistanceVoxelMap& map, const DistanceMapReference& ref)
	{
		return is_exact_distance_field(map, ref, expected_squared_distances(ref, map.dimensions()));
	}
}

#endif
