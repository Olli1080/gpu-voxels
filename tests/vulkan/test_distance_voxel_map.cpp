// Slice S5: Vulkan DistanceVoxelMap against the independent brute force reference (tests/reference).
#include "vulkan_fixture.h"

#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/DistanceVoxelMap.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>

#include "distance_map_support.h"
#include "prob_voxel_map_support.h"

using namespace gpu_voxels::vulkan;
using gvl_reference::DistanceMapReference;
using namespace gvl_test::distance;
using gvl_test::VulkanTest;
using namespace gvl_test::prob;

TEST_F(VulkanTest, DistanceMapRejectsInvalidConstruction)
{
	EXPECT_THROW(DistanceVoxelMap(ctx(), { 0, 4, 4 }, 1.0f), VulkanError);
	EXPECT_THROW(DistanceVoxelMap(ctx(), { 4, 4, 4 }, 0.0f), VulkanError);
	EXPECT_THROW(DistanceVoxelMap(ctx(), { 4, 4, 4 }, std::nanf("")), VulkanError);
	EXPECT_THROW(DistanceVoxelMap(ctx(), { 1024, 1, 1 }, 1.0f), VulkanError);
	EXPECT_THROW(DistanceVoxelMap(ctx(), { 1, 1024, 1 }, 1.0f), VulkanError);
	EXPECT_THROW(DistanceVoxelMap(ctx(), { 1, 1, 1024 }, 1.0f), VulkanError);
	DistanceVoxelMap ok(ctx(), { 1023, 1, 1 }, 1.0f);
	EXPECT_EQ(ok.voxel_count(), 1023u);
}

TEST_F(VulkanTest, DistanceMapStartsUninitialisedAndClears)
{
	DistanceVoxelMap map(ctx(), { 5, 4, 3 }, 1.0f);
	EXPECT_EQ(map.memory_usage_bytes(), 5u * 4u * 3u * 4u);
	for (uint32_t w : map.download_words())
		ASSERT_EQ(w, 0x3FFFFFFFu);
	EXPECT_EQ(map.squared_obstacle_distance({ 2, 2, 2 }), INT32_MAX);
	EXPECT_EQ(map.obstacle_distance({ 2, 2, 2 }), 46340);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 1, 1 } });
	map.clear_map();
	for (uint32_t w : map.download_words())
		ASSERT_EQ(w, 0x3FFFFFFFu);
	EXPECT_EQ(DistanceVoxelMap::pack({ 1, 2, 3 }), 1u | (2u << 10) | (3u << 20));
	const Coord3ui c = DistanceVoxelMap::unpack(DistanceVoxelMap::pack({ 1022, 7, 513 }));
	EXPECT_EQ(c.x, 1022u); EXPECT_EQ(c.y, 7u); EXPECT_EQ(c.z, 513u);
}

TEST_F(VulkanTest, DistanceMapInsertMarksObstaclesWithTheirOwnPosition)
{
	const Dim3ui dims{ 6, 5, 4 };
	DistanceVoxelMap map(ctx(), dims, 1.0f);
	const float nan = std::numeric_limits<float>::quiet_NaN();
	const std::vector<Point3f> points{ { 0.5f, 0.5f, 0.5f }, { 5.5f, 4.5f, 3.5f }, { -0.5f, 1.5f, 1.5f }, { 6.5f, 1.5f, 1.5f }, { nan, 0.5f, 0.5f },
		{ 2.5f, 2.5f, 2.5f } };
	EXPECT_EQ(map.insert_point_cloud(points), 3u);
	const std::vector<Coord3ui> coordinates{ { 3, 3, 3 }, { 6, 0, 0 }, { 0, 5, 0 }, { 0, 0, 4 }, { UINT32_MAX, 0, 0 } };
	EXPECT_EQ(map.insert_coordinate_list(coordinates), 4u);
	std::vector<std::vector<Point3f>> clouds{ { { 4.5f, 0.5f, 0.5f } }, {}, { { 1.5f, 3.5f, 2.5f }, { 99.0f, 0.0f, 0.0f } } };
	MetaPointCloud meta(ctx(), clouds);
	EXPECT_EQ(map.insert_meta_point_cloud(meta), 1u);

	const std::vector<Coord3ui> obstacles{ { 0, 0, 0 }, { 5, 4, 3 }, { 2, 2, 2 }, { 3, 3, 3 }, { 4, 0, 0 }, { 1, 3, 2 } };
	const auto words = map.download_words();
	uint32_t marked = 0;
	for (uint32_t i = 0; i < words.size(); ++i)
	{
		const Coord3ui p{ i % dims.x, (i / dims.x) % dims.y, i / dims.x / dims.y };
		bool is_obstacle = false;
		for (const auto& o : obstacles)
			is_obstacle = is_obstacle || (o.x == p.x && o.y == p.y && o.z == p.z);
		if (is_obstacle)
		{
			ASSERT_EQ(words[i], DistanceVoxelMap::pack(p));
			++marked;
		}
		else
			ASSERT_EQ(words[i], 0x3FFFFFFFu);
	}
	EXPECT_EQ(marked, 6u);

	for (uint8_t meaning : { uint8_t{ eBVM_FREE }, uint8_t{ eBVM_COLLISION }, uint8_t{ 40 } })
	{
		EXPECT_THROW(map.insert_point_cloud(points, meaning), VulkanError);
		EXPECT_THROW(map.insert_coordinate_list(coordinates, meaning), VulkanError);
		EXPECT_THROW(map.insert_meta_point_cloud(meta, meaning), VulkanError);
	}
}

TEST_F(VulkanTest, DistanceMapOfASingleObstacleIsTheEuclideanDistance)
{
	for (Dim3ui dims : { Dim3ui{ 9, 7, 5 }, Dim3ui{ 1, 1, 1 }, Dim3ui{ 1, 12, 1 }, Dim3ui{ 12, 1, 3 }, Dim3ui{ 2, 2, 2 }, Dim3ui{ 17, 3, 11 } })
		for (Coord3ui o : { Coord3ui{ 0, 0, 0 }, Coord3ui{ dims.x - 1, dims.y - 1, dims.z - 1 }, Coord3ui{ dims.x / 2, dims.y / 2, dims.z / 2 }, Coord3ui{ dims.x - 1, 0, dims.z / 2 } })
		{
			DistanceVoxelMap map(ctx(), dims, 1.0f);
			auto ref = distance_reference(dims);
			map.insert_coordinate_list(std::vector<Coord3ui>{ o });
			ref.insert_obstacle(o.x, o.y, o.z);
			map.compute_distances();
			ASSERT_TRUE(is_exact_distance_field(map, ref)) << dims.x << "x" << dims.y << "x" << dims.z << " obstacle " << o.x << "," << o.y << "," << o.z;
			for (const auto& s : map.download_obstacles())
				ASSERT_TRUE(s.x == o.x && s.y == o.y && s.z == o.z) << "with one obstacle it is the nearest for everybody";
		}
}

TEST_F(VulkanTest, DistanceMapMatchesBruteForceForRandomObstacles)
{
	std::mt19937 rng(81);
	const Dim3ui shapes[] = { { 23, 19, 17 }, { 31, 7, 40 }, { 1, 33, 29 }, { 64, 5, 3 }, { 8, 8, 8 }, { 2, 41, 2 } };
	for (Dim3ui dims : shapes)
		for (std::size_t count : { 1u, 2u, 5u, 40u, 400u })
		{
			DistanceVoxelMap map(ctx(), dims, 1.0f);
			auto ref = distance_reference(dims);
			const auto coordinates = random_coordinates(rng, count, dims);
			map.insert_coordinate_list(coordinates);
			ref.insert_coordinates(coordinates);
			map.compute_distances();
			ASSERT_TRUE(is_exact_distance_field(map, ref)) << dims.x << "x" << dims.y << "x" << dims.z << " obstacles " << count;
		}
}

TEST_F(VulkanTest, DistanceMapHandlesTiesAndFullMaps)
{
	// symmetric obstacles: many voxels have several nearest obstacles
	{
		const Dim3ui dims{ 11, 11, 11 };
		DistanceVoxelMap map(ctx(), dims, 1.0f);
		auto ref = distance_reference(dims);
		const std::vector<Coord3ui> coordinates{ { 0, 0, 0 }, { 10, 0, 0 }, { 0, 10, 0 }, { 0, 0, 10 }, { 10, 10, 0 }, { 10, 0, 10 }, { 0, 10, 10 }, { 10, 10, 10 } };
		map.insert_coordinate_list(coordinates);
		ref.insert_coordinates(coordinates);
		map.compute_distances();
		EXPECT_TRUE(is_exact_distance_field(map, ref));
		EXPECT_EQ(map.squared_obstacle_distance({ 5, 5, 5 }), 75); // 3 * 5^2 to every corner
	}
	// every voxel an obstacle: all distances 0 and every voxel stores itself
	{
		const Dim3ui dims{ 7, 6, 5 };
		DistanceVoxelMap map(ctx(), dims, 1.0f);
		std::vector<Coord3ui> all;
		for (uint32_t z = 0; z < dims.z; ++z)
			for (uint32_t y = 0; y < dims.y; ++y)
				for (uint32_t x = 0; x < dims.x; ++x)
					all.push_back({ x, y, z });
		map.insert_coordinate_list(all);
		map.compute_distances();
		const auto sites = map.download_obstacles();
		for (std::size_t i = 0; i < sites.size(); ++i)
			ASSERT_EQ(DistanceVoxelMap::pack(sites[i]), DistanceVoxelMap::pack(all[i]));
	}
}

TEST_F(VulkanTest, DistanceMapWithoutObstaclesStaysUninitialised)
{
	DistanceVoxelMap map(ctx(), { 9, 8, 7 }, 1.0f);
	map.compute_distances();
	for (uint32_t w : map.download_words())
		ASSERT_EQ(w, 0x3FFFFFFFu);
	EXPECT_EQ(map.squared_obstacle_distance({ 4, 4, 4 }), INT32_MAX);
}

TEST_F(VulkanTest, DistanceMapCanBeRecomputedAfterInsertingMoreObstacles)
{
	// CUDA needs a clear map between two runs; here obstacles are the voxels that store their own position, so every
	// other (stale) site is ignored
	std::mt19937 rng(82);
	const Dim3ui dims{ 19, 17, 13 };
	DistanceVoxelMap map(ctx(), dims, 1.0f);
	auto ref = distance_reference(dims);
	for (int round = 0; round < 4; ++round)
	{
		const auto coordinates = random_coordinates(rng, round == 0 ? 3 : 6, dims);
		map.insert_coordinate_list(coordinates);
		ref.insert_coordinates(coordinates);
		map.compute_distances();
		ASSERT_TRUE(is_exact_distance_field(map, ref)) << "round " << round;
		const auto first = map.download_words();
		map.compute_distances(); // idempotent
		EXPECT_EQ(map.download_words(), first) << "round " << round;
	}
	// clear_map forgets the obstacles
	map.clear_map();
	map.compute_distances();
	for (uint32_t w : map.download_words())
		ASSERT_EQ(w, 0x3FFFFFFFu);
}

TEST_F(VulkanTest, DistanceMapWorksAtTheMaximumAxisLength)
{
	// 1023 voxels: the longest line the shader supports; also the coordinate 1022 next to the uninitialised marker 1023
	std::mt19937 rng(83);
	for (Dim3ui dims : { Dim3ui{ 1023, 1, 1 }, Dim3ui{ 1, 1023, 1 }, Dim3ui{ 1, 1, 1023 }, Dim3ui{ 1023, 2, 3 }, Dim3ui{ 3, 1023, 2 } })
	{
		DistanceVoxelMap map(ctx(), dims, 1.0f);
		auto ref = distance_reference(dims);
		std::vector<Coord3ui> coordinates = random_coordinates(rng, 6, dims);
		coordinates.push_back({ dims.x - 1, dims.y - 1, dims.z - 1 });
		map.insert_coordinate_list(coordinates);
		ref.insert_coordinates(coordinates);
		map.compute_distances();
		ASSERT_TRUE(is_exact_distance_field(map, ref)) << dims.x << "x" << dims.y << "x" << dims.z;
	}
}

TEST_F(VulkanTest, DistanceMapQueriesFollowCuda)
{
	std::mt19937 rng(84);
	const Dim3ui dims{ 21, 15, 12 };
	DistanceVoxelMap map(ctx(), dims, 1.0f);
	auto ref = distance_reference(dims);
	const auto coordinates = random_coordinates(rng, 12, dims);
	map.insert_coordinate_list(coordinates);
	ref.insert_coordinates(coordinates);
	map.compute_distances();

	std::vector<uint32_t> indices;
	for (uint32_t i = 0; i < map.voxel_count(); i += 53)
		indices.push_back(i);
	indices.push_back(0);
	indices.push_back(static_cast<uint32_t>(map.voxel_count() - 1));
	indices.push_back(indices[3]); // duplicates are fine
	const auto squared = map.squared_distances(indices);
	const auto roots = map.distances(indices);
	ASSERT_EQ(squared.size(), indices.size());
	for (std::size_t k = 0; k < indices.size(); ++k)
	{
		const uint32_t i = indices[k];
		const Coord3ui p{ i % dims.x, (i / dims.x) % dims.y, i / dims.x / dims.y };
		const int64_t expected = ref.min_squared_distance(p.x, p.y, p.z);
		ASSERT_EQ(squared[k], expected) << i;
		ASSERT_EQ(roots[k], static_cast<int32_t>(std::floor(std::sqrt(static_cast<double>(expected))))) << i;
		ASSERT_EQ(map.squared_obstacle_distance(p), expected);
		ASSERT_EQ(map.obstacle_distance(p), roots[k]);
	}
	EXPECT_TRUE(map.squared_distances({}).empty());
	const std::vector<uint32_t> bad{ 0, static_cast<uint32_t>(map.voxel_count()) };
	EXPECT_THROW((void)map.squared_distances(bad), VulkanError);
	EXPECT_THROW((void)map.squared_obstacle_distance({ dims.x, 0, 0 }), VulkanError);
	EXPECT_THROW((void)map.obstacle_distance({ 0, dims.y, 0 }), VulkanError);
	// truncation: distance sqrt(2) -> 1, sqrt(5) -> 2, sqrt(8) -> 2
	DistanceVoxelMap small(ctx(), { 5, 5, 1 }, 1.0f);
	small.insert_coordinate_list(std::vector<Coord3ui>{ { 0, 0, 0 } });
	small.compute_distances();
	EXPECT_EQ(small.obstacle_distance({ 1, 1, 0 }), 1);
	EXPECT_EQ(small.obstacle_distance({ 2, 1, 0 }), 2);
	EXPECT_EQ(small.obstacle_distance({ 2, 2, 0 }), 2);
	EXPECT_EQ(small.obstacle_distance({ 3, 4, 0 }), 5);
}

TEST_F(VulkanTest, DistanceMapExtractsByteDistancesWithBorderRules)
{
	std::mt19937 rng(85);
	const Dim3ui dims{ 40, 33, 20 };
	DistanceVoxelMap map(ctx(), dims, 1.0f);
	auto ref = distance_reference(dims);
	const auto coordinates = random_coordinates(rng, 8, dims);
	map.insert_coordinate_list(coordinates);
	ref.insert_coordinates(coordinates);
	map.compute_distances();
	for (int radius : { 0, 1, 2, 5, 10, 25, 100 })
	{
		const auto bytes = map.extract_distances(radius);
		ASSERT_EQ(bytes.size(), map.voxel_count());
		for (uint32_t i = 0; i < bytes.size(); ++i)
		{
			const uint32_t x = i % dims.x, y = (i / dims.x) % dims.y, z = i / dims.x / dims.y;
			ASSERT_EQ(bytes[i], ref.extract(x, y, z, radius)) << "radius " << radius << " voxel " << x << "," << y << "," << z;
		}
	}
	// a map without obstacles: 127 away from the borders, border rules near them
	DistanceVoxelMap empty(ctx(), { 20, 20, 20 }, 1.0f);
	empty.compute_distances();
	const auto bytes = empty.extract_distances(3);
	EXPECT_EQ(bytes[(10 * 20 + 10) * 20 + 10], 127);
	EXPECT_EQ(bytes[(10 * 20 + 10) * 20 + 1], 1);
	EXPECT_EQ(bytes[(10 * 20 + 10) * 20 + 19], 1);
	EXPECT_EQ(bytes[(10 * 20 + 10) * 20 + 0], 0);
	// distances above 127 are capped
	DistanceVoxelMap far(ctx(), { 400, 1, 1 }, 1.0f);
	far.insert_coordinate_list(std::vector<Coord3ui>{ { 0, 0, 0 } });
	far.compute_distances();
	const auto capped = far.extract_distances(0);
	EXPECT_EQ(capped[126], 126);
	EXPECT_EQ(capped[127], 127);
	EXPECT_EQ(capped[128], 127);
	EXPECT_EQ(capped[399], 127);
}

TEST_F(VulkanTest, DistanceMapMergesOccupiedVoxelsOfAProbMap)
{
	std::mt19937 rng(86);
	const Dim3ui dims = kProbDims;
	ProbVoxelMap prob(ctx(), dims, kSide);
	auto prob_ref = prob_reference(dims);
	for (uint8_t meaning : { uint8_t{ eBVM_OCCUPIED }, uint8_t{ eBVM_FREE }, uint8_t{ 133 }, uint8_t{ 120 }, uint8_t{ eBVM_OCCUPIED } })
	{
		const auto points = random_points(rng, 300, dims);
		const int inc = meaning == eBVM_FREE ? -10 : (meaning == eBVM_OCCUPIED ? 72 : meaning - 129);
		prob.insert_point_cloud(points, meaning, UpdateOptions{ {}, UpdateMode::Sequential });
		prob_ref.apply_sequential(prob_ref.point_events(points, inc).events);
	}
	for (float threshold : { 0.5f, 0.6f, 0.75f, 0.9f, 0.0f, 1.0f })
	{
		DistanceVoxelMap map(ctx(), dims, kSide);
		auto ref = distance_reference(dims);
		map.merge_occupied(prob, threshold);
		const int level = ProbVoxelMapReference::probability_from_float(threshold);
		std::size_t expected = 0;
		for (uint64_t i = 0; i < map.voxel_count(); ++i)
			if (prob_ref.get(i) > level)
			{
				ref.insert_obstacle(static_cast<uint32_t>(i % dims.x), static_cast<uint32_t>((i / dims.x) % dims.y), static_cast<uint32_t>(i / dims.x / dims.y));
				++expected;
			}
		uint32_t marked = 0;
		const auto words = map.download_words();
		for (uint32_t i = 0; i < words.size(); ++i)
			marked += words[i] != 0x3FFFFFFFu ? 1 : 0;
		EXPECT_EQ(marked, expected) << "threshold " << threshold;
		if (expected > 0)
		{
			map.compute_distances();
			EXPECT_TRUE(is_exact_distance_field(map, ref)) << "threshold " << threshold;
		}
	}
	DistanceVoxelMap wrong(ctx(), { 5, 5, 5 }, kSide);
	EXPECT_THROW(wrong.merge_occupied(prob), VulkanError);
}

TEST_F(VulkanTest, DistanceMapOfALargeMapMatchesBruteForceAndReportsTime)
{
	std::mt19937 rng(87);
	const Dim3ui dims{ 128, 96, 80 };
	DistanceVoxelMap map(ctx(), dims, 1.0f);
	auto ref = distance_reference(dims);
	const auto coordinates = random_coordinates(rng, 300, dims);
	map.insert_coordinate_list(coordinates);
	ref.insert_coordinates(coordinates);
	const auto t0 = std::chrono::steady_clock::now();
	map.compute_distances();
	const auto t1 = std::chrono::steady_clock::now();
	std::cout << "[perf] distance transform of " << dims.x << "x" << dims.y << "x" << dims.z << " (" << map.voxel_count() << " voxels, "
		<< ref.obstacle_count() << " obstacles): " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
	EXPECT_TRUE(is_exact_distance_field(map, ref));
}
