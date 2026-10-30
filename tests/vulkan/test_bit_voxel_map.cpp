// Slice S1: Vulkan BitVoxelMap against the independent CPU reference (tests/reference).
#include "vulkan_fixture.h"

#include <algorithm>
#include <bit>
#include <bitset>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>

#include "../reference/bit_voxel_map_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_reference::BitVoxelMapReference;
using gvl_test::VulkanTest;

namespace
{
	constexpr Dim3ui kDims{ 37, 29, 41 }; // deliberately not powers of two
	constexpr float kSide = 0.1f;

	BitVoxelMapReference make_reference(Dim3ui d, float side) { return BitVoxelMapReference(d.x, d.y, d.z, side); }

	/// Random point strictly inside a voxel (fraction 0.1..0.9 of the voxel edge: no boundary cases).
	Point3f random_inside_point(std::mt19937& rng, Dim3ui d, float side)
	{
		std::uniform_int_distribution<uint32_t> vx(0, d.x - 1), vy(0, d.y - 1), vz(0, d.z - 1);
		std::uniform_real_distribution<float> frac(0.1f, 0.9f);
		return { (static_cast<float>(vx(rng)) + frac(rng)) * side, (static_cast<float>(vy(rng)) + frac(rng)) * side,
			(static_cast<float>(vz(rng)) + frac(rng)) * side };
	}

	std::vector<Point3f> random_points(std::mt19937& rng, std::size_t n, Dim3ui d, float side)
	{
		std::vector<Point3f> points(n);
		for (auto& p : points)
			p = random_inside_point(rng, d, side);
		return points;
	}

	/// Compares the whole map (or a range) with the reference and reports the first differing voxel.
	::testing::AssertionResult equal_to_reference(const BitVoxelMap& map, const BitVoxelMapReference& ref,
		uint64_t first = 0, uint64_t count = UINT64_MAX)
	{
		count = std::min(count, map.voxel_count() - first);
		const auto gpu = map.download_words(first, count);
		const auto cpu = ref.to_words(first, count);
		if (gpu.size() != cpu.size())
			return ::testing::AssertionFailure() << "size mismatch " << gpu.size() << " vs " << cpu.size();
		for (std::size_t i = 0; i < gpu.size(); ++i)
		{
			if (gpu[i] != cpu[i])
			{
				const uint64_t voxel = first + i / BitVoxelMap::kWordsPerVoxel;
				return ::testing::AssertionFailure() << "voxel " << voxel << " word " << i % BitVoxelMap::kWordsPerVoxel
					<< ": vulkan 0x" << std::hex << gpu[i] << " reference 0x" << cpu[i];
			}
		}
		return ::testing::AssertionSuccess();
	}
}

TEST_F(VulkanTest, BitMapStartsEmptyAndReportsGeometry)
{
	BitVoxelMap map(ctx(), kDims, kSide);
	EXPECT_EQ(map.voxel_count(), 37u * 29u * 41u);
	EXPECT_EQ(map.memory_usage_bytes(), 37u * 29u * 41u * 32u);
	EXPECT_TRUE(equal_to_reference(map, make_reference(kDims, kSide)));
}

TEST_F(VulkanTest, BitMapRejectsInvalidConstruction)
{
	EXPECT_THROW(BitVoxelMap(ctx(), { 0, 4, 4 }, 1.0f), VulkanError);
	EXPECT_THROW(BitVoxelMap(ctx(), { 4, 4, 4 }, 0.0f), VulkanError);
	EXPECT_THROW(BitVoxelMap(ctx(), { 4, 4, 4 }, -1.0f), VulkanError);
	EXPECT_THROW(BitVoxelMap(ctx(), { 4, 4, 4 }, std::nanf("")), VulkanError);
	EXPECT_THROW(BitVoxelMap(ctx(), { (1u << 24) + 1, 1, 1 }, 1.0f), VulkanError);
	EXPECT_THROW(BitVoxelMap(ctx(), { 4096, 4096, 4096 }, 1.0f), VulkanError); // 2 TiB
}

TEST_F(VulkanTest, BitMapInsertsPointsAtKnownVoxels)
{
	BitVoxelMap map(ctx(), { 4, 3, 2 }, 1.0f);
	// voxel (x,y,z) -> index z*12 + y*4 + x
	const std::vector<Point3f> points{ { 0.5f, 0.5f, 0.5f }, { 3.5f, 2.5f, 1.5f }, { 1.99f, 0.01f, 1.0f } };
	EXPECT_EQ(map.insert_point_cloud(points, eBVM_OCCUPIED), 0u);

	const auto words = map.download_words();
	auto bit = [&](uint64_t voxel, uint32_t b) { return (words[voxel * 8 + b / 32] >> (b % 32)) & 1u; };
	EXPECT_EQ(bit(0, 1), 1u);
	EXPECT_EQ(bit(1 * 12 + 2 * 4 + 3, 1), 1u);  // (3,2,1)
	EXPECT_EQ(bit(1 * 12 + 0 * 4 + 1, 1), 1u);  // (1,0,1)
	uint32_t total_bits = 0;
	for (uint32_t w : words)
		total_bits += static_cast<uint32_t>(std::popcount(w));
	EXPECT_EQ(total_bits, 3u);
}

TEST_F(VulkanTest, BitMapMeaningsAcrossWordBoundariesAreIndependent)
{
	BitVoxelMap map(ctx(), { 2, 2, 2 }, 1.0f);
	auto ref = make_reference({ 2, 2, 2 }, 1.0f);
	const std::vector<Point3f> points{ { 0.5f, 0.5f, 0.5f } };
	for (uint8_t meaning : { uint8_t{ 0 }, uint8_t{ 1 }, uint8_t{ 31 }, uint8_t{ 32 }, uint8_t{ 63 }, uint8_t{ 64 },
		uint8_t{ 127 }, uint8_t{ 128 }, uint8_t{ 224 }, uint8_t{ 255 } })
	{
		map.insert_point_cloud(points, meaning);
		ref.insert_points(points, meaning);
	}
	EXPECT_TRUE(equal_to_reference(map, ref));
}

TEST_F(VulkanTest, BitMapPointCloudMatchesReference)
{
	for (const float side : { 0.0625f, 0.1f })
	{
		std::mt19937 rng(1234);
		BitVoxelMap map(ctx(), kDims, side);
		auto ref = make_reference(kDims, side);

		auto points = random_points(rng, 100'000, kDims, side);
		// Points outside the map and invalid values.
		const float inf = std::numeric_limits<float>::infinity();
		const float extent_x = static_cast<float>(kDims.x) * side;
		const std::vector<Point3f> outside{ { -0.5f * side, 0.5f * side, 0.5f * side }, { 0.5f * side, -2.0f * side, 0.5f * side },
			{ 0.5f * side, 0.5f * side, -0.01f * side }, { extent_x + 0.5f * side, 0.5f * side, 0.5f * side },
			{ std::nanf(""), 0.5f * side, 0.5f * side }, { inf, 0.5f * side, 0.5f * side }, { -inf, 0.0f, 0.0f },
			{ 1e30f, 1e30f, 1e30f } };
		points.insert(points.end(), outside.begin(), outside.end());

		const uint32_t expected_outside = ref.insert_points(points, eBVM_OCCUPIED);
		EXPECT_EQ(expected_outside, outside.size());
		EXPECT_EQ(map.insert_point_cloud(points, eBVM_OCCUPIED), expected_outside) << "side " << side;
		EXPECT_TRUE(equal_to_reference(map, ref)) << "side " << side;

		// A second cloud with another meaning accumulates (OR) in the same voxels.
		auto more = random_points(rng, 50'000, kDims, side);
		ref.insert_points(more, 77);
		map.insert_point_cloud(more, 77);
		EXPECT_TRUE(equal_to_reference(map, ref)) << "side " << side;
	}
}

TEST_F(VulkanTest, BitMapManyPointsInOneVoxelContendForTheSameWord)
{
	BitVoxelMap map(ctx(), { 8, 8, 8 }, 1.0f);
	auto ref = make_reference({ 8, 8, 8 }, 1.0f);
	std::vector<Point3f> points(1'000'000, Point3f{ 3.25f, 4.5f, 5.75f });
	for (uint8_t meaning : { uint8_t{ 5 }, uint8_t{ 6 }, uint8_t{ 200 } })
	{
		map.insert_point_cloud(points, meaning);
		ref.insert_points(points, meaning);
	}
	EXPECT_TRUE(equal_to_reference(map, ref));
}

TEST_F(VulkanTest, BitMapCoordinateListMatchesReference)
{
	std::mt19937 rng(99);
	BitVoxelMap map(ctx(), kDims, kSide);
	auto ref = make_reference(kDims, kSide);
	std::uniform_int_distribution<uint32_t> x(0, kDims.x + 3), y(0, kDims.y + 3), z(0, kDims.z + 3); // some out of range
	std::vector<Coord3ui> coords(60'000);
	for (auto& c : coords)
		c = { x(rng), y(rng), z(rng) };
	coords.push_back({ UINT32_MAX, 0, 0 });

	const uint32_t expected_outside = ref.insert_coordinates(coords, 9);
	EXPECT_GT(expected_outside, 0u);
	EXPECT_EQ(map.insert_coordinate_list(coords, 9), expected_outside);
	EXPECT_TRUE(equal_to_reference(map, ref));
}

TEST_F(VulkanTest, BitMapEmptyInputsAreNoOps)
{
	BitVoxelMap map(ctx(), kDims, kSide);
	EXPECT_EQ(map.insert_point_cloud({}, eBVM_OCCUPIED), 0u);
	EXPECT_EQ(map.insert_coordinate_list({}, eBVM_OCCUPIED), 0u);
	EXPECT_TRUE(equal_to_reference(map, make_reference(kDims, kSide)));
}

TEST_F(VulkanTest, BitMapClearAndClearMeaning)
{
	std::mt19937 rng(5);
	BitVoxelMap map(ctx(), kDims, kSide);
	auto ref = make_reference(kDims, kSide);
	for (uint8_t meaning : { uint8_t{ 1 }, uint8_t{ 40 }, uint8_t{ 41 }, uint8_t{ 255 } })
	{
		const auto points = random_points(rng, 20'000, kDims, kSide);
		map.insert_point_cloud(points, meaning);
		ref.insert_points(points, meaning);
	}
	ASSERT_TRUE(equal_to_reference(map, ref));

	map.clear_bit_voxel_meaning(40);
	ref.clear_meaning(40);
	EXPECT_TRUE(equal_to_reference(map, ref));
	map.clear_bit_voxel_meaning(40); // clearing an absent meaning changes nothing
	EXPECT_TRUE(equal_to_reference(map, ref));

	map.clear_map();
	ref.clear();
	EXPECT_TRUE(equal_to_reference(map, ref));
}

TEST_F(VulkanTest, BitMapCollisionMatchesReference)
{
	std::mt19937 rng(777);
	BitVoxelMap a(ctx(), kDims, kSide), b(ctx(), kDims, kSide);
	auto ref_a = make_reference(kDims, kSide), ref_b = make_reference(kDims, kSide);

	// a: occupied (1), another meaning (200) and "free only" voxels (0); b: occupied (5) and free only (0).
	for (const auto& [meaning, n] : { std::pair<uint8_t, int>{ 1, 8000 }, { 200, 3000 }, { 0, 6000 } })
	{
		const auto pts = random_points(rng, n, kDims, kSide);
		a.insert_point_cloud(pts, meaning);
		ref_a.insert_points(pts, meaning);
	}
	for (const auto& [meaning, n] : { std::pair<uint8_t, int>{ 5, 9000 }, { 0, 6000 } })
	{
		const auto pts = random_points(rng, n, kDims, kSide);
		b.insert_point_cloud(pts, meaning);
		ref_b.insert_points(pts, meaning);
	}
	ASSERT_TRUE(equal_to_reference(a, ref_a));
	ASSERT_TRUE(equal_to_reference(b, ref_b));

	const uint32_t expected = ref_a.collide_with(ref_b);
	ASSERT_GT(expected, 0u) << "test data must contain collisions";
	EXPECT_EQ(a.collide_with(b), expected);
	EXPECT_TRUE(equal_to_reference(a, ref_a)) << "colliding voxels of the first map must carry the collision bit";
	EXPECT_TRUE(equal_to_reference(b, ref_b)) << "the second map must stay unchanged";
	EXPECT_EQ(a.collide_with(b), expected) << "collision count is stable (collision bit counts as occupied)";
}

TEST_F(VulkanTest, BitMapFreeBitAloneIsNotOccupied)
{
	BitVoxelMap a(ctx(), { 4, 4, 4 }, 1.0f), b(ctx(), { 4, 4, 4 }, 1.0f);
	const std::vector<Point3f> p{ { 0.5f, 0.5f, 0.5f } };
	a.insert_point_cloud(p, eBVM_FREE);
	b.insert_point_cloud(p, eBVM_OCCUPIED);
	EXPECT_EQ(a.collide_with(b), 0u);  // a only has the FREE bit
	EXPECT_EQ(b.collide_with(a), 0u);

	a.insert_point_cloud(p, eBVM_UNKNOWN); // any other bit makes the voxel occupied
	EXPECT_EQ(a.collide_with(b), 1u);
	EXPECT_EQ(b.collide_with(a), 1u);
}

TEST_F(VulkanTest, BitMapCollisionRejectsInvalidArguments)
{
	BitVoxelMap a(ctx(), kDims, kSide), b(ctx(), { kDims.x + 1, kDims.y, kDims.z }, kSide);
	EXPECT_THROW(a.collide_with(a), VulkanError);
	EXPECT_THROW(a.collide_with(b), VulkanError);
	EXPECT_THROW((void)a.download_words(a.voxel_count() + 1, 1), VulkanError);
}

// 256^3 voxels = 512 MiB per map; sparse content, full comparison of the colliding map.
TEST_F(VulkanTest, BitMapLargeMap256Cubed)
{
	constexpr Dim3ui dims{ 256, 256, 256 };
	constexpr uint64_t bytes = 256ull * 256 * 256 * 32;
	if (ctx().limits().maxStorageBufferRange < bytes || ctx().info().device_local_heap_bytes < 4 * bytes)
		GTEST_SKIP() << "device too small for two 512 MiB maps";

	std::mt19937 rng(2026);
	const float side = 0.5f;
	BitVoxelMap a(ctx(), dims, side), b(ctx(), dims, side);
	auto ref_a = make_reference(dims, side), ref_b = make_reference(dims, side);

	const auto shared = random_points(rng, 3000, dims, side); // present in both maps => collisions
	for (auto* target : { &a, &b })
		target->insert_point_cloud(shared, eBVM_OCCUPIED);
	ref_a.insert_points(shared, eBVM_OCCUPIED);
	ref_b.insert_points(shared, eBVM_OCCUPIED);
	const auto only_a = random_points(rng, 20'000, dims, side);
	const auto only_b = random_points(rng, 20'000, dims, side);
	a.insert_point_cloud(only_a, 17);
	ref_a.insert_points(only_a, 17);
	b.insert_point_cloud(only_b, 18);
	ref_b.insert_points(only_b, 18);

	const auto t0 = std::chrono::steady_clock::now();
	const uint32_t collisions = a.collide_with(b);
	const auto t1 = std::chrono::steady_clock::now();
	const uint32_t expected = ref_a.collide_with(ref_b);
	std::cout << "[vulkan] collide 2 x 512 MiB maps: " << std::chrono::duration<double, std::milli>(t1 - t0).count()
		<< " ms, " << collisions << " collisions (" << ctx().info().name << ")" << std::endl;
	EXPECT_EQ(collisions, expected);
	EXPECT_GT(expected, 3000u);

	// Full comparison in slices of 1M voxels.
	for (uint64_t first = 0; first < a.voxel_count(); first += (1u << 20))
		ASSERT_TRUE(equal_to_reference(a, ref_a, first, 1u << 20)) << "slice at voxel " << first;
}

// ---- collide_with_types (CUDA collideWithTypes / SVCollider) ----

namespace
{
	std::vector<uint32_t> to_words(const std::bitset<256>& bits)
	{
		std::vector<uint32_t> words(8, 0);
		for (uint32_t b = 0; b < 256; ++b)
			if (bits.test(b))
				words[b / 32] |= 1u << (b % 32);
		return words;
	}
}

TEST_F(VulkanTest, BitMapCollideWithTypesMatchesReference)
{
	std::mt19937 rng(4242);
	BitVoxelMap a(ctx(), kDims, kSide), b(ctx(), kDims, kSide);
	auto ref_a = make_reference(kDims, kSide), ref_b = make_reference(kDims, kSide);

	// Both maps hold plain OCCUPIED voxels (not a "type" collision), plus swept-volume ids; ids overlap partially.
	const std::vector<std::pair<uint8_t, int>> meanings_a{ { 1, 6000 }, { 4, 1500 }, { 20, 1500 }, { 100, 1500 }, { 254, 800 }, { 0, 3000 } };
	const std::vector<std::pair<uint8_t, int>> meanings_b{ { 1, 6000 }, { 4, 1500 }, { 21, 1500 }, { 100, 1500 }, { 255, 800 }, { 3, 2000 } };
	for (const auto& [m, n] : meanings_a)
	{
		const auto pts = random_points(rng, n, kDims, kSide);
		a.insert_point_cloud(pts, m);
		ref_a.insert_points(pts, m);
	}
	for (const auto& [m, n] : meanings_b)
	{
		const auto pts = random_points(rng, n, kDims, kSide);
		b.insert_point_cloud(pts, m);
		ref_b.insert_points(pts, m);
	}

	const auto expected = ref_a.collide_with_types(ref_b);
	ASSERT_GT(expected.count, 0u);
	const auto result = a.collide_with_types(b);
	EXPECT_EQ(result.count, expected.count);
	EXPECT_EQ(std::vector<uint32_t>(result.types.begin(), result.types.end()), to_words(expected.types));
	EXPECT_TRUE(equal_to_reference(a, ref_a));
	EXPECT_TRUE(equal_to_reference(b, ref_b));
	// The shared swept-volume ids 4 and 100 must be reported, the non-shared ones (20/21, 254/255) must not.
	EXPECT_TRUE(expected.types.test(4));
	EXPECT_TRUE(expected.types.test(100));
	EXPECT_FALSE(expected.types.test(20));
	EXPECT_FALSE(expected.types.test(254));
}

TEST_F(VulkanTest, BitMapCollideWithTypesIgnoresLowBits)
{
	BitVoxelMap a(ctx(), { 4, 4, 4 }, 1.0f), b(ctx(), { 4, 4, 4 }, 1.0f);
	const std::vector<Point3f> p{ { 0.5f, 0.5f, 0.5f } };
	for (uint8_t low : { eBVM_FREE, eBVM_OCCUPIED, eBVM_COLLISION, eBVM_UNKNOWN })
	{
		a.insert_point_cloud(p, low);
		b.insert_point_cloud(p, low);
	}
	const auto none = a.collide_with_types(b);
	EXPECT_EQ(none.count, 0u); // bits 0..3 shared, but they are not types
	EXPECT_EQ(a.collide_with(b), 1u); // the default collider still sees two occupied voxels

	a.insert_point_cloud(p, 4);
	b.insert_point_cloud(p, 4);
	const auto one = a.collide_with_types(b);
	EXPECT_EQ(one.count, 1u);
	EXPECT_EQ(one.types[0], 1u << 4);
	for (std::size_t w = 1; w < one.types.size(); ++w)
		EXPECT_EQ(one.types[w], 0u);
}

TEST_F(VulkanTest, BitMapCollideWithTypesRejectsInvalidArguments)
{
	BitVoxelMap a(ctx(), kDims, kSide), b(ctx(), { kDims.x, kDims.y + 1, kDims.z }, kSide);
	EXPECT_THROW((void)a.collide_with_types(a), VulkanError);
	EXPECT_THROW((void)a.collide_with_types(b), VulkanError);
}
