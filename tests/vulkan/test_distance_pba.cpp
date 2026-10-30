// Slice S5: the Parallel Banding Algorithm of the Vulkan DistanceVoxelMap against the brute force reference and against the
// exact separable algorithm.
#include "vulkan_fixture.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/DistanceVoxelMap.h>

#include "distance_map_support.h"

using namespace gpu_voxels::vulkan;
using gvl_reference::DistanceMapReference;
using namespace gvl_test::distance;
using gvl_test::VulkanTest;

namespace
{
	std::vector<int32_t> all_squared_distances(const DistanceVoxelMap& map)
	{
		std::vector<uint32_t> indices(map.voxel_count());
		for (uint32_t i = 0; i < indices.size(); ++i)
			indices[i] = i;
		return map.squared_distances(indices);
	}

	std::string describe(Dim3ui d, const ParallelBandingOptions& o)
	{
		return std::to_string(d.x) + "x" + std::to_string(d.y) + "x" + std::to_string(d.z) + " m1=" + std::to_string(o.m1) + " m2=" +
			std::to_string(o.m2) + " m3=" + std::to_string(o.m3);
	}
}

TEST_F(VulkanTest, DistancePbaMatchesBruteForceWithASingleBand)
{
	std::mt19937 rng(91);
	const Dim3ui shapes[] = { { 23, 19, 17 }, { 31, 7, 40 }, { 1, 33, 29 }, { 64, 5, 3 }, { 8, 8, 8 }, { 2, 41, 2 }, { 1, 1, 1 }, { 5, 1, 9 } };
	for (Dim3ui dims : shapes)
		for (std::size_t count : { 1u, 2u, 5u, 40u, 400u })
		{
			DistanceVoxelMap map(ctx(), dims, 1.0f);
			auto ref = distance_reference(dims);
			const auto coordinates = random_coordinates(rng, count, dims);
			map.insert_coordinate_list(coordinates);
			ref.insert_coordinates(coordinates);
			map.compute_distances(DistanceAlgorithm::ParallelBanding);
			ASSERT_TRUE(is_exact_distance_field(map, ref)) << describe(dims, {}) << " obstacles " << count;
		}
}

TEST_F(VulkanTest, DistancePbaMatchesBruteForceWithBands)
{
	std::mt19937 rng(92);
	struct Case { Dim3ui dims; ParallelBandingOptions options; };
	const Case cases[] = {
		{ { 32, 32, 32 }, { 2, 1, 1 } }, { { 32, 32, 32 }, { 4, 2, 2 } }, { { 32, 32, 32 }, { 8, 4, 4 } }, { { 32, 32, 32 }, { 16, 8, 8 } },
		{ { 32, 32, 32 }, { 1, 32, 16 } }, { { 32, 32, 32 }, { 2, 2, 16 } },
		{ { 48, 24, 30 }, { 3, 2, 3 } }, { { 48, 24, 30 }, { 5, 4, 6 } }, { { 48, 24, 30 }, { 15, 8, 8 } },
		{ { 40, 16, 12 }, { 4, 4, 4 } }, { { 16, 40, 12 }, { 6, 8, 2 } }, { { 64, 8, 20 }, { 10, 8, 8 } }, { { 12, 12, 50 }, { 25, 4, 12 } },
	};
	for (const Case& c : cases)
		for (std::size_t count : { 1u, 2u, 5u, 40u, 300u })
		{
			DistanceVoxelMap map(ctx(), c.dims, 1.0f);
			auto ref = distance_reference(c.dims);
			const auto coordinates = random_coordinates(rng, count, c.dims);
			map.insert_coordinate_list(coordinates);
			ref.insert_coordinates(coordinates);
			map.compute_distances(DistanceAlgorithm::ParallelBanding, c.options);
			ASSERT_TRUE(is_exact_distance_field(map, ref)) << describe(c.dims, c.options) << " obstacles " << count;
		}
}

TEST_F(VulkanTest, DistancePbaHandlesEmptyBandsAndSpecialObstacleLayouts)
{
	const Dim3ui dims{ 32, 32, 32 };
	const ParallelBandingOptions options{ 8, 8, 8 }; // 4 voxel bands: most bands of most lines are empty
	std::vector<std::vector<Coord3ui>> layouts;
	layouts.push_back({ { 0, 0, 0 } });
	layouts.push_back({ { 31, 31, 31 } });
	layouts.push_back({ { 0, 0, 0 }, { 31, 31, 31 } });
	layouts.push_back({ { 15, 16, 17 } });
	{
		std::vector<Coord3ui> plane; // a full plane: every line along z has exactly one obstacle
		for (uint32_t x = 0; x < 32; ++x)
			for (uint32_t y = 0; y < 32; ++y)
				plane.push_back({ x, y, 13 });
		layouts.push_back(plane);
	}
	{
		std::vector<Coord3ui> wall; // a full wall in x: every line along x is hit at one position
		for (uint32_t y = 0; y < 32; ++y)
			for (uint32_t z = 0; z < 32; ++z)
				wall.push_back({ 7, y, z });
		layouts.push_back(wall);
	}
	{
		std::vector<Coord3ui> diagonal;
		for (uint32_t i = 0; i < 32; ++i)
			diagonal.push_back({ i, i, i });
		layouts.push_back(diagonal);
	}
	{
		std::vector<Coord3ui> corners;
		for (uint32_t i = 0; i < 8; ++i)
			corners.push_back({ (i & 1) * 31u, ((i >> 1) & 1) * 31u, ((i >> 2) & 1) * 31u });
		layouts.push_back(corners);
	}
	for (std::size_t l = 0; l < layouts.size(); ++l)
		for (ParallelBandingOptions o : { ParallelBandingOptions{}, options })
		{
			DistanceVoxelMap map(ctx(), dims, 1.0f);
			auto ref = distance_reference(dims);
			map.insert_coordinate_list(layouts[l]);
			ref.insert_coordinates(layouts[l]);
			map.compute_distances(DistanceAlgorithm::ParallelBanding, o);
			ASSERT_TRUE(is_exact_distance_field(map, ref)) << "layout " << l << " " << describe(dims, o);
		}
}

TEST_F(VulkanTest, DistancePbaAgreesWithTheExactAlgorithm)
{
	std::mt19937 rng(93);
	const Dim3ui dims{ 40, 24, 36 };
	for (ParallelBandingOptions o : { ParallelBandingOptions{ 1, 1, 1 }, ParallelBandingOptions{ 4, 4, 4 }, ParallelBandingOptions{ 9, 2, 8 } })
	{
		DistanceVoxelMap pba(ctx(), dims, 1.0f), exact(ctx(), dims, 1.0f);
		const auto coordinates = random_coordinates(rng, 150, dims);
		pba.insert_coordinate_list(coordinates);
		exact.insert_coordinate_list(coordinates);
		pba.compute_distances(DistanceAlgorithm::ParallelBanding, o);
		exact.compute_distances(DistanceAlgorithm::Exact);
		EXPECT_EQ(all_squared_distances(pba), all_squared_distances(exact)) << describe(dims, o);
	}
}

TEST_F(VulkanTest, DistancePbaCanBeRecomputedAndMixedWithTheExactAlgorithm)
{
	std::mt19937 rng(94);
	const Dim3ui dims{ 32, 16, 20 };
	const ParallelBandingOptions options{ 4, 2, 4 };
	DistanceVoxelMap map(ctx(), dims, 1.0f);
	auto ref = distance_reference(dims);
	for (int round = 0; round < 4; ++round)
	{
		const auto coordinates = random_coordinates(rng, 8, dims);
		map.insert_coordinate_list(coordinates);
		ref.insert_coordinates(coordinates);
		map.compute_distances(round % 2 == 0 ? DistanceAlgorithm::ParallelBanding : DistanceAlgorithm::Exact, options);
		ASSERT_TRUE(is_exact_distance_field(map, ref)) << "round " << round;
	}
	const auto before = all_squared_distances(map);
	map.compute_distances(DistanceAlgorithm::ParallelBanding, options);
	EXPECT_TRUE(is_exact_distance_field(map, ref));
	EXPECT_EQ(all_squared_distances(map), before) << "repeating the computation changes nothing";
	map.clear_map();
	map.compute_distances(DistanceAlgorithm::ParallelBanding, options);
	for (uint32_t w : map.download_words())
		ASSERT_EQ(w, 0x3FFFFFFFu);
}

TEST_F(VulkanTest, DistancePbaWorksAtTheMaximumAxisLength)
{
	std::mt19937 rng(95);
	for (Dim3ui dims : { Dim3ui{ 1023, 1, 1 }, Dim3ui{ 1, 1023, 1 }, Dim3ui{ 1, 1, 1023 }, Dim3ui{ 1023, 3, 2 }, Dim3ui{ 3, 1023, 2 }, Dim3ui{ 3, 3, 1023 } })
	{
		DistanceVoxelMap map(ctx(), dims, 1.0f);
		auto ref = distance_reference(dims);
		std::vector<Coord3ui> coordinates = random_coordinates(rng, 6, dims);
		coordinates.push_back({ dims.x - 1, dims.y - 1, dims.z - 1 });
		map.insert_coordinate_list(coordinates);
		ref.insert_coordinates(coordinates);
		map.compute_distances(DistanceAlgorithm::ParallelBanding);
		ASSERT_TRUE(is_exact_distance_field(map, ref)) << describe(dims, {});
	}
}

TEST_F(VulkanTest, DistancePbaRejectsInvalidBandCounts)
{
	DistanceVoxelMap map(ctx(), { 32, 24, 30 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 3, 3, 3 } });
	const auto words = map.download_words();
	auto run = [&](ParallelBandingOptions o) { map.compute_distances(DistanceAlgorithm::ParallelBanding, o); };
	EXPECT_THROW(run({ 0, 1, 1 }), VulkanError);   // m1 = 0
	EXPECT_THROW(run({ 4, 1, 1 }), VulkanError);   // 4 does not divide dim_z = 30
	EXPECT_THROW(run({ 31, 1, 1 }), VulkanError);  // larger than dim_z
	EXPECT_THROW(run({ 30, 1, 1 }), VulkanError);  // divides dim_z but leaves one layer per band
	EXPECT_THROW(run({ 1, 0, 1 }), VulkanError);   // m2 = 0
	EXPECT_THROW(run({ 1, 3, 1 }), VulkanError);   // not a power of two
	EXPECT_THROW(run({ 1, 16, 1 }), VulkanError);  // 16 does not divide dim_y = 24
	EXPECT_THROW(run({ 1, 1, 0 }), VulkanError);   // m3 = 0
	EXPECT_THROW(run({ 1, 1, 32 }), VulkanError);  // more than 16
	EXPECT_THROW(run({ 1, 1, 5 }), VulkanError);   // 5 does not divide dim_x = 32
	EXPECT_EQ(map.download_words(), words) << "rejected calls must not change the map";
	// the cap of 16 invocations per line is independent of the dimensions
	DistanceVoxelMap cube(ctx(), { 32, 32, 32 }, 1.0f);
	EXPECT_THROW(cube.compute_distances(DistanceAlgorithm::ParallelBanding, { 1, 1, 32 }), VulkanError);
	EXPECT_NO_THROW(cube.compute_distances(DistanceAlgorithm::ParallelBanding, { 16, 32, 16 }));
	EXPECT_NO_THROW(run({ 15, 8, 8 }));
}

TEST_F(VulkanTest, DistancePbaPerformanceReport)
{
	std::mt19937 rng(96);
	const Dim3ui dims{ 128, 96, 80 };
	const auto coordinates = random_coordinates(rng, 300, dims);
	struct Setup { const char* name; DistanceAlgorithm algorithm; ParallelBandingOptions options; };
	const Setup setups[] = { { "Exact", DistanceAlgorithm::Exact, {} }, { "PBA m=1/1/1", DistanceAlgorithm::ParallelBanding, { 1, 1, 1 } },
		{ "PBA m=4/4/4", DistanceAlgorithm::ParallelBanding, { 4, 4, 4 } }, { "PBA m=8/8/8", DistanceAlgorithm::ParallelBanding, { 8, 8, 8 } },
		{ "PBA m=16/32/16", DistanceAlgorithm::ParallelBanding, { 16, 32, 16 } } };
	std::vector<int32_t> first;
	for (const Setup& s : setups)
	{
		DistanceVoxelMap map(ctx(), dims, 1.0f);
		map.insert_coordinate_list(coordinates);
		const auto t0 = std::chrono::steady_clock::now();
		map.compute_distances(s.algorithm, s.options);
		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "[perf] distance transform " << dims.x << "x" << dims.y << "x" << dims.z << ", " << s.name << ": "
			<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
		const auto squared = all_squared_distances(map);
		if (first.empty())
			first = squared;
		else
			EXPECT_EQ(squared, first) << s.name;
	}
}

TEST_F(VulkanTest, DistancePbaIsExactForEveryValidBandCombination)
{
	// m1 = 32 would mean one layer per band, which is rejected (see DistancePbaRejectsInvalidBandCounts)
	std::mt19937 rng(97);
	const Dim3ui dims{ 32, 32, 32 };
	DistanceVoxelMap map(ctx(), dims, 1.0f); // one map: the pipelines are created once
	for (std::size_t count : { 5u, 60u })
	{
		const auto coordinates = random_coordinates(rng, count, dims);
		auto ref = distance_reference(dims);
		ref.insert_coordinates(coordinates);
		const auto expected = expected_squared_distances(ref, dims);
		for (uint32_t m1 : { 1u, 2u, 4u, 8u, 16u })
			for (uint32_t m2 : { 1u, 2u, 4u, 8u, 16u, 32u })
				for (uint32_t m3 : { 1u, 2u, 4u, 8u, 16u })
				{
					map.clear_map();
					map.insert_coordinate_list(coordinates);
					map.compute_distances(DistanceAlgorithm::ParallelBanding, { m1, m2, m3 });
					ASSERT_TRUE(is_exact_distance_field(map, ref, expected)) << describe(dims, { m1, m2, m3 }) << " obstacles " << count;
				}
	}
}

TEST_F(VulkanTest, DistancePbaKeepsObstaclesInTallMapsWithBands)
{
	// CUDA compares |1023 - z| for uninitialised voxels in phase 1 (flood, inter-band steps); in maps with z + distance > 1023
	// this replaces valid obstacles by "uninitialised". The port treats uninitialised voxels as infinitely far.
	const Dim3ui dims{ 1, 1, 1022 };
	auto ref = distance_reference(dims);
	for (uint32_t obstacle_z : { 520u, 3u, 1018u })
		for (uint32_t m1 : { 1u, 2u, 7u, 73u })
		{
			DistanceVoxelMap map(ctx(), dims, 1.0f);
			map.insert_coordinate_list(std::vector<Coord3ui>{ { 0, 0, obstacle_z } });
			ref.clear();
			ref.insert_obstacle(0, 0, obstacle_z);
			map.compute_distances(DistanceAlgorithm::ParallelBanding, { m1, 1, 1 });
			ASSERT_TRUE(is_exact_distance_field(map, ref)) << "obstacle at z=" << obstacle_z << " m1=" << m1;
		}
}
