// Slice S4: Vulkan voxel lists (BitVoxelList, CountingVoxelList) against the independent CPU reference (tests/reference).
#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/VoxelList.h>

#include "voxel_list_support.h"

using namespace gpu_voxels::vulkan;
using gvl_reference::BitVoxelMapReference;
using gvl_reference::VoxelListReference;
using gvl_test::VulkanTest;

using namespace gvl_test::lists;

TEST_F(VulkanTest, ListRejectsInvalidConstruction)
{
	EXPECT_THROW(BitVoxelList(ctx(), { 0, 4, 4 }, 1.0f), VulkanError);
	EXPECT_THROW(CountingVoxelList(ctx(), { 4, 4, 0 }, 1.0f), VulkanError);
	EXPECT_THROW(BitVoxelList(ctx(), { 4, 4, 4 }, 0.0f), VulkanError);
	EXPECT_THROW(BitVoxelList(ctx(), { 4, 4, 4 }, std::nanf("")), VulkanError);
	EXPECT_THROW(BitVoxelList(ctx(), { (1u << 24) + 1, 1, 1 }, 1.0f), VulkanError);
	EXPECT_THROW(BitVoxelList(ctx(), { 4096, 4096, 4096 }, 1.0f), VulkanError); // ids would need 36 bits
}

TEST_F(VulkanTest, ListStartsEmptyAndClears)
{
	BitVoxelList list(ctx(), kDims, kSide);
	EXPECT_TRUE(list.empty());
	EXPECT_EQ(list.size(), 0u);
	EXPECT_EQ(list.memory_usage_bytes(), 0u);
	EXPECT_TRUE(list.download_ids().empty());
	EXPECT_TRUE(list.download_bit_vectors().empty());
	EXPECT_EQ(list.reference_dimensions(), kDims);
	EXPECT_EQ(list.voxel_side_length(), kSide);

	std::mt19937 rng(1);
	list.insert_point_cloud(random_points(rng, 100), 5);
	EXPECT_FALSE(list.empty());
	EXPECT_EQ(list.memory_usage_bytes(), static_cast<uint64_t>(list.size()) * (4 + 12 + 32));
	list.clear();
	EXPECT_TRUE(list.empty());
	EXPECT_TRUE(equal_to_reference(list, bit_reference()));
	list.make_unique(); // no-op on an empty list
	EXPECT_TRUE(list.empty());
}

TEST_F(VulkanTest, BitListInsertsKnownVoxels)
{
	BitVoxelList list(ctx(), { 4, 3, 2 }, 1.0f);
	// voxel (x,y,z) -> id (z*3 + y)*4 + x
	const std::vector<Point3f> first{ { 0.5f, 0.5f, 0.5f }, { 3.5f, 2.5f, 1.5f } };
	EXPECT_EQ(list.insert_point_cloud(first, eBVM_OCCUPIED), 0u);
	const std::vector<Point3f> second{ { 0.9f, 0.1f, 0.2f } }; // same voxel as the first point
	EXPECT_EQ(list.insert_point_cloud(second, 40), 0u);

	EXPECT_EQ(list.download_ids(), (std::vector<uint32_t>{ 0, 23 }));
	const auto coords = list.download_coordinates();
	ASSERT_EQ(coords.size(), 2u);
	EXPECT_EQ(coords[0].x, 0u); EXPECT_EQ(coords[0].y, 0u); EXPECT_EQ(coords[0].z, 0u);
	EXPECT_EQ(coords[1].x, 3u); EXPECT_EQ(coords[1].y, 2u); EXPECT_EQ(coords[1].z, 1u);
	const auto vectors = list.download_bit_vectors();
	BitVector256 expected0{}; expected0[0] = 1u << 1; expected0[1] = 1u << 8; // meanings 1 and 40
	BitVector256 expected1{}; expected1[0] = 1u << 1;
	EXPECT_EQ(vectors[0], expected0);
	EXPECT_EQ(vectors[1], expected1);
}

TEST_F(VulkanTest, ListMergesSmallGroupsOfDuplicates)
{
	// two, three and four entries of the same voxel in one insert; the list is empty before, so exactly 2 / 3 / 4 entries are merged
	for (std::size_t copies : { 2u, 3u, 4u })
	{
		BitVoxelList bits(ctx(), { 4, 4, 4 }, 1.0f);
		CountingVoxelList counts(ctx(), { 4, 4, 4 }, 1.0f);
		const std::vector<Point3f> points(copies, Point3f{ 2.5f, 1.5f, 0.5f });
		bits.insert_point_cloud(points, 9);
		counts.insert_point_cloud(points, 9);
		EXPECT_EQ(bits.download_ids(), (std::vector<uint32_t>{ 6 })) << copies;
		EXPECT_EQ(counts.download_counts(), (std::vector<int8_t>{ static_cast<int8_t>(copies) })) << copies;
	}
	BitVoxelList bits(ctx(), { 4, 4, 4 }, 1.0f);
	bits.insert_point_cloud(std::vector<Point3f>{ { 2.5f, 1.5f, 0.5f }, { 2.6f, 1.6f, 0.6f } }, 9);
	bits.insert_point_cloud(std::vector<Point3f>{ { 0.5f, 0.5f, 0.5f } }, 10); // 1 existing + 1 new: no duplicates
	EXPECT_EQ(bits.download_ids(), (std::vector<uint32_t>{ 0, 6 }));
}

TEST_F(VulkanTest, BitListMatchesReferenceForRandomBatches)
{
	BitVoxelList list(ctx(), kDims, kSide);
	auto ref = bit_reference();
	std::mt19937 rng(11);
	for (int batch = 0; batch < 6; ++batch)
	{
		const auto points = random_points(rng, 20000);
		const uint8_t meaning = random_meaning(rng);
		EXPECT_EQ(list.insert_point_cloud(points, meaning), ref.insert_points(points, meaning));
		ASSERT_TRUE(equal_to_reference(list, ref)) << "batch " << batch;
	}
	// sorted by id, ids unique, coordinates consistent with the ids
	const auto ids = list.download_ids();
	const auto coords = list.download_coordinates();
	for (std::size_t i = 0; i < ids.size(); ++i)
	{
		if (i > 0)
			ASSERT_LT(ids[i - 1], ids[i]);
		ASSERT_EQ(ids[i], (coords[i].z * kDims.y + coords[i].y) * kDims.x + coords[i].x);
	}
}

TEST_F(VulkanTest, BitListDropsPointsOutsideTheReferenceMap)
{
	BitVoxelList list(ctx(), kDims, kSide);
	auto ref = bit_reference();
	const float nan = std::numeric_limits<float>::quiet_NaN();
	const float inf = std::numeric_limits<float>::infinity();
	const std::vector<Point3f> points{
		{ 0.55f, 0.55f, 0.55f },  // inside
		{ -1.0f, 0.5f, 0.5f },    // negative x
		{ 0.5f, -0.5f, 0.5f },    // negative y
		{ 0.5f, 0.5f, -3.0f },    // negative z
		{ 1000.0f, 0.5f, 0.5f },  // too large
		{ 0.5f, 1000.0f, 0.5f },
		{ 0.5f, 0.5f, 1000.0f },
		{ nan, 0.5f, 0.5f },
		{ 0.5f, nan, 0.5f },
		{ 0.5f, 0.5f, nan },
		{ inf, 0.5f, 0.5f },
		{ 0.5f, -inf, 0.5f },
		{ 3.65f, 2.75f, 4.05f },  // last voxel (36, 27, 40)
		{ 3.75f, 0.55f, 0.55f },  // first voxel beyond the map along x (37)
		{ 0.55f, 2.95f, 0.55f },  // ... along y (29)
		{ 0.55f, 0.55f, 4.15f },  // ... along z (41)
		{ -0.05f, 0.55f, 0.55f }, // voxel -1 along x
		{ 0.55f, -0.05f, 0.55f }, // voxel -1 along y
		{ 0.55f, 0.55f, -0.05f }, // voxel -1 along z
	};
	const uint32_t dropped = list.insert_point_cloud(points, eBVM_OCCUPIED);
	EXPECT_EQ(dropped, 17u);
	EXPECT_EQ(dropped, ref.insert_points(points, eBVM_OCCUPIED));
	EXPECT_EQ(list.size(), 2u);
	EXPECT_TRUE(equal_to_reference(list, ref));

	// everything outside: list unchanged
	const std::vector<Point3f> outside{ { -1, -1, -1 }, { 500, 0, 0 } };
	EXPECT_EQ(list.insert_point_cloud(outside, 3), 2u);
	EXPECT_TRUE(equal_to_reference(list, ref));
}

TEST_F(VulkanTest, BitListInsertsCoordinateLists)
{
	BitVoxelList list(ctx(), kDims, kSide);
	auto ref = bit_reference();
	std::mt19937 rng(12);
	std::uniform_int_distribution<uint32_t> vx(0, kDims.x + 3), vy(0, kDims.y + 3), vz(0, kDims.z + 3); // some outside
	std::vector<Coord3ui> coordinates;
	for (int i = 0; i < 5000; ++i)
		coordinates.push_back({ vx(rng), vy(rng), vz(rng) });
	coordinates.push_back({ kDims.x - 1, kDims.y - 1, kDims.z - 1 });
	coordinates.push_back({ kDims.x, 0, 0 });
	coordinates.push_back({ UINT32_MAX, 0, 0 });
	coordinates.push_back({ 0, UINT32_MAX, UINT32_MAX });
	const uint32_t dropped = list.insert_coordinate_list(coordinates, 77);
	EXPECT_EQ(dropped, ref.insert_coordinates(coordinates, 77));
	EXPECT_GT(dropped, 100u);
	EXPECT_TRUE(equal_to_reference(list, ref));
}

TEST_F(VulkanTest, BitListInsertsMetaPointClouds)
{
	std::mt19937 rng(13);
	const std::vector<std::vector<Point3f>> clouds{ random_points(rng, 3000), {}, random_points(rng, 1500), random_points(rng, 1) };
	MetaPointCloud meta(ctx(), clouds);

	{
		BitVoxelList list(ctx(), kDims, kSide);
		auto ref = bit_reference();
		EXPECT_EQ(list.insert_meta_point_cloud(meta, 9), ref.insert_meta_clouds(clouds, { 9, 9, 9, 9 }));
		EXPECT_TRUE(equal_to_reference(list, ref));
	}
	{
		BitVoxelList list(ctx(), kDims, kSide);
		auto ref = bit_reference();
		const std::vector<uint8_t> meanings{ 10, 11, 200, 33 };
		EXPECT_EQ(list.insert_meta_point_cloud(meta, meanings), ref.insert_meta_clouds(clouds, meanings));
		EXPECT_TRUE(equal_to_reference(list, ref));
		EXPECT_THROW(list.insert_meta_point_cloud(meta, std::vector<uint8_t>{ 1, 2 }), VulkanError);
	}
	{
		// an empty cloud inserts nothing
		MetaPointCloud empty(ctx(), std::vector<uint32_t>{ 0, 0 });
		BitVoxelList list(ctx(), kDims, kSide);
		EXPECT_EQ(list.insert_meta_point_cloud(empty, 1), 0u);
		EXPECT_TRUE(list.empty());
	}
}

TEST_F(VulkanTest, BitListMetaCloudPointsOutsideAreDropped)
{
	const std::vector<std::vector<Point3f>> clouds{ { { 0.55f, 0.55f, 0.55f }, { -5.0f, 0.0f, 0.0f } }, { { 100.0f, 0.0f, 0.0f } } };
	MetaPointCloud meta(ctx(), clouds);
	BitVoxelList list(ctx(), kDims, kSide);
	auto ref = bit_reference();
	const std::vector<uint8_t> meanings{ 4, 5 };
	EXPECT_EQ(list.insert_meta_point_cloud(meta, meanings), 2u);
	ref.insert_meta_clouds(clouds, meanings);
	EXPECT_TRUE(equal_to_reference(list, ref));
	EXPECT_EQ(list.size(), 1u);
}

TEST_F(VulkanTest, CountingListCountsAndWrapsLikeInt8)
{
	CountingVoxelList list(ctx(), { 4, 4, 4 }, 1.0f);
	const std::vector<Point3f> hundred_thirty(130, Point3f{ 1.5f, 1.5f, 1.5f });
	EXPECT_EQ(list.insert_point_cloud(hundred_thirty, 7), 0u);
	ASSERT_EQ(list.size(), 1u);
	EXPECT_EQ(list.download_counts(), (std::vector<int8_t>{ -126 })); // 130 wraps to -126
	EXPECT_EQ(list.download_voxel_words(), (std::vector<uint32_t>{ 0xFFFFFF82u })); // sign extended
	const std::vector<Point3f> more(126, Point3f{ 1.5f, 1.5f, 1.5f });
	list.insert_point_cloud(more, 200); // the meaning is ignored by counting lists
	EXPECT_EQ(list.size(), 1u);
	EXPECT_EQ(list.download_counts(), (std::vector<int8_t>{ 0 })); // 256 wraps to 0, the entry stays
	list.insert_point_cloud(std::vector<Point3f>(3, Point3f{ 1.5f, 1.5f, 1.5f }), 1);
	EXPECT_EQ(list.download_counts(), (std::vector<int8_t>{ 3 }));
}

TEST_F(VulkanTest, CountingListMatchesReferenceForRandomBatches)
{
	const Dim3ui dims{ 10, 10, 10 }; // few voxels: counts exceed 127 and wrap
	CountingVoxelList list(ctx(), dims, kSide);
	auto ref = counting_reference(dims);
	std::mt19937 rng(21);
	for (int batch = 0; batch < 6; ++batch)
	{
		const auto points = random_points(rng, 30000, dims);
		EXPECT_EQ(list.insert_point_cloud(points, random_meaning(rng)), ref.insert_points(points, 0));
		ASSERT_TRUE(equal_to_reference(list, ref)) << "batch " << batch;
	}
	const auto counts = list.download_counts();
	EXPECT_TRUE(std::any_of(counts.begin(), counts.end(), [](int8_t c) { return c < 0; })) << "test must reach the wrap-around";
}

TEST_F(VulkanTest, CountingListInsertsCoordinatesAndMetaClouds)
{
	CountingVoxelList list(ctx(), kDims, kSide);
	auto ref = counting_reference();
	std::mt19937 rng(22);
	std::uniform_int_distribution<uint32_t> vx(0, 5), vy(0, 5), vz(0, 5);
	std::vector<Coord3ui> coordinates;
	for (int i = 0; i < 4000; ++i)
		coordinates.push_back({ vx(rng), vy(rng), vz(rng) });
	coordinates.push_back({ 1000, 0, 0 });
	EXPECT_EQ(list.insert_coordinate_list(coordinates, 0), ref.insert_coordinates(coordinates, 0));
	EXPECT_TRUE(equal_to_reference(list, ref));

	const std::vector<std::vector<Point3f>> clouds{ random_points(rng, 2000, { 6, 6, 6 }), random_points(rng, 2000, { 6, 6, 6 }) };
	MetaPointCloud meta(ctx(), clouds);
	EXPECT_EQ(list.insert_meta_point_cloud(meta, 3), ref.insert_meta_clouds(clouds, { 3, 3 }));
	EXPECT_TRUE(equal_to_reference(list, ref));
}

TEST_F(VulkanTest, CountingListRemoveUnderpopulated)
{
	const Dim3ui dims{ 6, 6, 6 };
	std::mt19937 rng(23);
	const auto points = random_points(rng, 4000, dims);
	for (int threshold : { -128, -5, 0, 1, 2, 10, 18, 19, 20, 21, 25, 40, 127 })
	{
		CountingVoxelList list(ctx(), dims, kSide);
		auto ref = counting_reference(dims);
		list.insert_point_cloud(points, 0);
		ref.insert_points(points, 0);
		list.remove_underpopulated(static_cast<int8_t>(threshold));
		ref.remove_underpopulated(threshold);
		EXPECT_TRUE(equal_to_reference(list, ref)) << "threshold " << threshold;
	}

	// negative counts (wrapped) are removed by threshold 0 and kept by threshold -128
	CountingVoxelList list(ctx(), { 4, 4, 4 }, 1.0f);
	list.insert_point_cloud(std::vector<Point3f>(200, Point3f{ 0.5f, 0.5f, 0.5f }), 0);      // 200 -> -56
	list.insert_point_cloud(std::vector<Point3f>(5, Point3f{ 3.5f, 3.5f, 3.5f }), 0);        // 5
	list.remove_underpopulated(0);
	EXPECT_EQ(list.download_ids(), (std::vector<uint32_t>{ 63 }));
	EXPECT_EQ(list.download_counts(), (std::vector<int8_t>{ 5 }));
	list.remove_underpopulated(6);
	EXPECT_TRUE(list.empty());
	list.remove_underpopulated(0); // on an empty list
	EXPECT_TRUE(list.empty());

}
