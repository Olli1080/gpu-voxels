// Slice S4: list operations (collision, subtraction, map interplay, scale) against the CPU reference.
#include "vulkan_fixture.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>

#include "voxel_list_support.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
using namespace gvl_test::lists;

namespace
{
	/// Two overlapping random point sets in a small volume (so that many ids coincide) and the lists / references built from them.
	struct ListPair
	{
		BitVoxelList a;
		BitVoxelList b;
		CountingVoxelList c;
		VoxelListReference ref_a;
		VoxelListReference ref_b;
		VoxelListReference ref_c;
	};

	ListPair make_lists(Context& context, uint32_t seed, Dim3ui dims = { 12, 12, 12 })
	{
		ListPair p{ BitVoxelList(context, dims, kSide), BitVoxelList(context, dims, kSide), CountingVoxelList(context, dims, kSide),
			bit_reference(dims), bit_reference(dims), counting_reference(dims) };
		std::mt19937 rng(seed);
		const auto points_a = random_points(rng, 700, dims);
		const auto points_b = random_points(rng, 500, dims);
		const auto points_c = random_points(rng, 1500, dims);
		p.a.insert_point_cloud(points_a, 3);
		p.ref_a.insert_points(points_a, 3);
		p.b.insert_point_cloud(points_b, 70);
		p.ref_b.insert_points(points_b, 70);
		p.c.insert_point_cloud(points_c, 0);
		p.ref_c.insert_points(points_c, 0);
		return p;
	}
}

TEST_F(VulkanTest, BitListCollidesWithBitListByIds)
{
	auto p = make_lists(ctx(), 31);
	const uint32_t expected = p.ref_a.collide_with(p.ref_b);
	EXPECT_GT(expected, 50u);
	EXPECT_LT(expected, p.ref_a.size());
	EXPECT_EQ(p.a.collide_with(p.b), expected);
	EXPECT_EQ(p.b.collide_with(p.a), p.ref_b.collide_with(p.ref_a));
	EXPECT_EQ(p.a.collide_with(p.a), p.a.size()); // a list collides with itself everywhere
	// no side effects
	EXPECT_TRUE(equal_to_reference(p.a, p.ref_a));
	EXPECT_TRUE(equal_to_reference(p.b, p.ref_b));
}

TEST_F(VulkanTest, ListCollisionWithDisjointAndEmptyListsIsZero)
{
	const Dim3ui dims{ 12, 12, 12 };
	BitVoxelList low(ctx(), dims, kSide), high(ctx(), dims, kSide), empty(ctx(), dims, kSide);
	low.insert_coordinate_list(std::vector<Coord3ui>{ { 0, 0, 0 }, { 1, 0, 0 }, { 2, 2, 2 } }, 1);
	high.insert_coordinate_list(std::vector<Coord3ui>{ { 11, 11, 11 }, { 5, 5, 5 } }, 1);
	EXPECT_EQ(low.collide_with(high), 0u);
	EXPECT_EQ(low.collide_with(empty), 0u);
	EXPECT_EQ(empty.collide_with(low), 0u);
	high.insert_coordinate_list(std::vector<Coord3ui>{ { 2, 2, 2 } }, 1);
	EXPECT_EQ(low.collide_with(high), 1u);
	EXPECT_EQ(high.collide_with(low), 1u);
}

TEST_F(VulkanTest, CountingListCollidesWithBitListAboveThreshold)
{
	auto p = make_lists(ctx(), 32);
	for (int threshold : { -128, -1, 0, 1, 2, 3, 4, 6, 8, 127 })
	{
		const uint32_t expected = p.ref_c.collide_with(p.ref_b, threshold);
		EXPECT_EQ(p.c.collide_with(p.b, static_cast<int8_t>(threshold)), expected) << "threshold " << threshold;
	}
	EXPECT_GT(p.ref_c.collide_with(p.ref_b, 2), 0u);
	EXPECT_LT(p.ref_c.collide_with(p.ref_b, 2), p.ref_c.collide_with(p.ref_b, 1)) << "the threshold must matter in this test";
	EXPECT_TRUE(equal_to_reference(p.c, p.ref_c));
}

TEST_F(VulkanTest, BitListSubtractRemovesCommonEntries)
{
	auto p = make_lists(ctx(), 33);
	p.a.subtract(p.b);
	p.ref_a.subtract(p.ref_b);
	EXPECT_TRUE(equal_to_reference(p.a, p.ref_a));
	EXPECT_EQ(p.a.collide_with(p.b), 0u);
	EXPECT_TRUE(equal_to_reference(p.b, p.ref_b)); // other list untouched

	p.a.subtract(p.b); // nothing left to remove
	EXPECT_TRUE(equal_to_reference(p.a, p.ref_a));
	BitVoxelList empty(ctx(), { 12, 12, 12 }, kSide);
	p.a.subtract(empty);
	EXPECT_TRUE(equal_to_reference(p.a, p.ref_a));
	empty.subtract(p.b);
	EXPECT_TRUE(empty.empty());
	p.b.subtract(p.b); // self subtraction empties the list
	EXPECT_TRUE(p.b.empty());
}

TEST_F(VulkanTest, CountingListSubtractVariants)
{
	{
		auto p = make_lists(ctx(), 34);
		p.c.subtract_from_counting_list(p.b);
		p.ref_c.subtract(p.ref_b);
		EXPECT_TRUE(equal_to_reference(p.c, p.ref_c));
	}
	{
		auto p = make_lists(ctx(), 35);
		CountingVoxelList other(ctx(), { 12, 12, 12 }, kSide);
		auto ref_other = counting_reference({ 12, 12, 12 });
		std::mt19937 rng(36);
		const auto points = random_points(rng, 300, { 12, 12, 12 });
		other.insert_point_cloud(points, 0);
		ref_other.insert_points(points, 0);
		p.c.subtract(other);
		p.ref_c.subtract(ref_other);
		EXPECT_TRUE(equal_to_reference(p.c, p.ref_c));
	}
}

TEST_F(VulkanTest, ListOperationsRejectMismatchingReferenceDimensions)
{
	BitVoxelList a(ctx(), { 12, 12, 12 }, kSide), b(ctx(), { 12, 12, 13 }, kSide);
	CountingVoxelList c(ctx(), { 12, 12, 12 }, kSide);
	EXPECT_THROW((void)a.collide_with(b), VulkanError);
	EXPECT_THROW(a.subtract(b), VulkanError);
	EXPECT_THROW((void)c.collide_with(b, 1), VulkanError);
	EXPECT_THROW(c.subtract_from_counting_list(b), VulkanError);
	BitVoxelMap wrong_map(ctx(), { 12, 12, 13 }, kSide);
	EXPECT_THROW(a.collide_with(wrong_map), VulkanError);
	EXPECT_THROW(a.insert_occupied_voxels(wrong_map), VulkanError);
}

TEST_F(VulkanTest, BitListCollisionWithMapMarksCollidingEntries)
{
	const Dim3ui dims{ 14, 13, 12 };
	std::mt19937 rng(41);
	BitVoxelMap map(ctx(), dims, kSide);
	auto ref_map = map_reference(dims);
	const auto occupied_points = random_points(rng, 600, dims);
	map.insert_point_cloud(occupied_points, eBVM_OCCUPIED);
	ref_map.insert_points(occupied_points, eBVM_OCCUPIED);
	const auto free_points = random_points(rng, 600, dims); // voxels that only have the FREE bit are not occupied
	map.insert_point_cloud(free_points, eBVM_FREE);
	ref_map.insert_points(free_points, eBVM_FREE);
	const auto swept_points = random_points(rng, 200, dims);
	map.insert_point_cloud(swept_points, 99);
	ref_map.insert_points(swept_points, 99);

	BitVoxelList list(ctx(), dims, kSide);
	auto ref = bit_reference(dims);
	for (uint8_t meaning : { uint8_t{ 4 }, uint8_t{ 0 } }) // meaning 0: list voxels without any bit except FREE are inspected as well
	{
		const auto points = random_points(rng, 900, dims);
		list.insert_point_cloud(points, meaning);
		ref.insert_points(points, meaning);
	}
	const uint32_t expected = ref.collide_with_map(ref_map);
	EXPECT_GT(expected, 100u);
	EXPECT_LT(expected, ref.size());
	EXPECT_EQ(list.collide_with(map), expected);
	EXPECT_TRUE(equal_to_reference(list, ref)); // collision bits are set on the same entries

	// repeating the check finds the same collisions (the marker bit does not change the outcome)
	EXPECT_EQ(list.collide_with(map), expected);
	EXPECT_TRUE(equal_to_reference(list, ref));
	EXPECT_EQ(BitVoxelList(ctx(), dims, kSide).collide_with(map), 0u);
}

TEST_F(VulkanTest, MapMergeInsertsListCoordinatesAsOccupied)
{
	const Dim3ui dims{ 14, 13, 12 };
	std::mt19937 rng(42);
	BitVoxelList list(ctx(), dims, kSide);
	const auto points = random_points(rng, 800, dims);
	list.insert_point_cloud(points, 0);        // payload is ignored by the merge, even FREE-only entries
	list.insert_point_cloud(random_points(rng, 300, dims), 50);

	BitVoxelMap map(ctx(), dims, kSide);
	auto ref_map = map_reference(dims);
	const auto existing = random_points(rng, 200, dims);
	map.insert_point_cloud(existing, 17);
	ref_map.insert_points(existing, 17);

	std::vector<Coord3ui> coordinates = list.download_coordinates();
	EXPECT_EQ(map.merge_voxel_list(list), 0u);
	ref_map.insert_coordinates(coordinates, eBVM_OCCUPIED);
	const auto gpu = map.download_words();
	EXPECT_EQ(gpu, ref_map.to_words(0, map.voxel_count()));

	// merging an empty list changes nothing; a smaller map counts the coordinates outside it
	EXPECT_EQ(map.merge_voxel_list(BitVoxelList(ctx(), dims, kSide)), 0u);
	EXPECT_EQ(map.download_words(), gpu);
	BitVoxelMap small(ctx(), { 7, 7, 7 }, kSide);
	auto ref_small = map_reference({ 7, 7, 7 });
	const uint32_t outside = small.merge_voxel_list(list);
	EXPECT_EQ(outside, ref_small.insert_coordinates(coordinates, eBVM_OCCUPIED));
	EXPECT_GT(outside, 0u);
	EXPECT_EQ(small.download_words(), ref_small.to_words(0, small.voxel_count()));
}

TEST_F(VulkanTest, ListTakesOccupiedVoxelsOfAMap)
{
	const Dim3ui dims{ 14, 13, 12 };
	std::mt19937 rng(43);
	BitVoxelMap map(ctx(), dims, kSide);
	auto ref_map = map_reference(dims);
	for (uint8_t meaning : { uint8_t{ 1 }, uint8_t{ 0 }, uint8_t{ 5 }, uint8_t{ 200 }, uint8_t{ 1 } })
	{
		const auto points = random_points(rng, 400, dims);
		map.insert_point_cloud(points, meaning);
		ref_map.insert_points(points, meaning);
	}

	BitVoxelList list(ctx(), dims, kSide);
	auto ref = bit_reference(dims);
	const auto prior = random_points(rng, 300, dims);
	list.insert_point_cloud(prior, 7); // OR-ed with the map voxels where they overlap
	ref.insert_points(prior, 7);

	uint32_t occupied = 0;
	for (const auto& [index, voxel] : ref_map.voxels())
		occupied += BitVoxelMapReference::occupied(voxel) ? 1 : 0;
	EXPECT_GT(occupied, 300u);
	EXPECT_EQ(list.insert_occupied_voxels(map), occupied);
	EXPECT_EQ(ref.insert_occupied_voxels(ref_map), occupied);
	EXPECT_TRUE(equal_to_reference(list, ref));

	// an empty map adds nothing
	BitVoxelMap empty_map(ctx(), dims, kSide);
	EXPECT_EQ(list.insert_occupied_voxels(empty_map), 0u);
	EXPECT_TRUE(equal_to_reference(list, ref));
}

TEST_F(VulkanTest, BitListHandlesAMillionPoints)
{
	const Dim3ui dims{ 128, 128, 128 };
	std::mt19937 rng(51);
	const auto points = random_points(rng, 1'000'000, dims);
	BitVoxelList list(ctx(), dims, kSide);
	auto ref = bit_reference(dims);

	const auto t0 = std::chrono::steady_clock::now();
	EXPECT_EQ(list.insert_point_cloud(points, 12), 0u);
	const auto t1 = std::chrono::steady_clock::now();
	ref.insert_points(points, 12);
	std::cout << "[perf] 1M points -> " << list.size() << " entries in "
		<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms (includes upload, sort, merge)\n";
	EXPECT_TRUE(equal_to_reference(list, ref));
}

TEST_F(VulkanTest, CountingListInsertsMoreThanOneUploadChunk)
{
	// kMaxItemsPerUpload is 4 Mi points: this input needs two chunks, the second one merges into the first result
	const Dim3ui dims{ 64, 64, 64 };
	std::mt19937 rng(52);
	const auto points = random_points(rng, (4u << 20) + 1000, dims);
	CountingVoxelList list(ctx(), dims, kSide);
	auto ref = counting_reference(dims);
	EXPECT_EQ(list.insert_point_cloud(points, 0), 0u);
	ref.insert_points(points, 0);
	EXPECT_TRUE(equal_to_reference(list, ref));
}
