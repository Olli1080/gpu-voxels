// K34: the element limits of the primitives, the octree build and the voxel lists. Each limit is tested exactly at its value (must work)
// and one above (must throw a VulkanError), so a boundary that is off by one in either direction is found.
//  * scan:  Primitives::kMaxElements = 2^24
//  * sort:  Primitives::kMaxSortElements = 65535 x 256 (one work group per tile of 256 elements); octree build and voxel list entries
//    are bound by the sort.
#include "vulkan_fixture.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/Primitives.h>
#include <gpu_voxels/vulkan/VoxelList.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;

namespace
{
	/// The call must throw a VulkanError whose message contains `part`: the check of the class itself, not the one of a primitive
	/// further down (which would also stop a call that is one element over the limit).
	template <class F>
	void expect_error_containing(F&& call, const std::string& part)
	{
		try
		{
			call();
			ADD_FAILURE() << "no exception, expected one containing: " << part;
		}
		catch (const VulkanError& e)
		{
			EXPECT_NE(std::string(e.what()).find(part), std::string::npos) << "message: " << e.what() << " (expected: " << part << ")";
		}
	}

	/// The i-th voxel of the cube of side 256 in x-fastest order.
	Coord3ui cube_voxel(uint32_t i) { return { i & 255u, (i >> 8) & 255u, i >> 16 }; }
}

TEST(SizeLimits, ConstantsDescribeTheSortAndTheScan)
{
	static_assert(Primitives::kMaxSortElements == 65535u * 256u);
	static_assert(Primitives::kMaxSortElements < Primitives::kMaxElements);
	static_assert(Octree::kMaxVoxels == Primitives::kMaxSortElements);
	static_assert(VoxelListBase::kMaxEntries == Primitives::kMaxSortElements);
	EXPECT_EQ(Primitives::kMaxSortElements, 16'776'960u);
	EXPECT_EQ(Primitives::kMaxElements, 16'777'216u);
}

TEST_F(VulkanTest, ScanAcceptsTheLimitAndRejectsOneMore)
{
	Primitives primitives(ctx());
	std::vector<uint32_t> input(Primitives::kMaxElements, 1u);
	const auto result = primitives.exclusive_scan(input);
	ASSERT_EQ(result.size(), input.size());
	for (std::size_t i = 0; i < result.size(); i += 4093)
		ASSERT_EQ(result[i], static_cast<uint32_t>(i));
	EXPECT_EQ(result.back(), Primitives::kMaxElements - 1);

	input.push_back(1u);
	EXPECT_THROW((void)primitives.exclusive_scan(input), VulkanError);
}

TEST_F(VulkanTest, SortAcceptsTheLimitAndRejectsOneMore)
{
	Primitives primitives(ctx());
	std::mt19937 rng(34);
	const uint32_t n = Primitives::kMaxSortElements;
	std::vector<uint32_t> keys(n), values(n);
	for (auto& k : keys)
		k = rng() & ((1u << 20) - 1);
	std::iota(values.begin(), values.end(), 0u);
	const auto original = keys;
	primitives.sort_pairs(keys, values, 20);
	EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
	for (std::size_t i = 0; i < keys.size(); i += 4099)
		ASSERT_EQ(original[values[i]], keys[i]) << "value must still belong to its key";

	keys.push_back(0);
	values.push_back(0);
	EXPECT_THROW(primitives.sort_pairs(keys, values, 20), VulkanError);
	std::vector<uint64_t> keys64(n + 1, 0);
	EXPECT_THROW(primitives.sort_pairs64(keys64, values, 42), VulkanError);
}

TEST_F(VulkanTest, OctreeBuildAcceptsTheLimitAndRejectsOneMore)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const uint32_t n = Octree::kMaxVoxels;
	Octree tree(ctx(), OctreeKind::Deterministic);

	// build from coordinates: the first n voxels of a cube of side 256 (all distinct)
	std::vector<OctreeVoxel> voxels(n);
	for (uint32_t i = 0; i < n; ++i)
	{
		const Coord3ui c = cube_voxel(i);
		voxels[i] = { c.x, c.y, c.z };
	}
	tree.build(voxels);
	EXPECT_EQ(tree.voxel_count(), n);
	EXPECT_EQ(tree.dropped_count(), 0u);
	voxels.push_back(voxels.front());
	expect_error_containing([&] { tree.build(voxels); }, "too many voxels");
	voxels.clear();
	voxels.shrink_to_fit();

	// build from Morton ids: the ids 0 .. n-1 (all distinct and valid)
	std::vector<uint64_t> ids(n);
	std::iota(ids.begin(), ids.end(), uint64_t(0));
	tree.build_from_ids(ids);
	EXPECT_EQ(tree.voxel_count(), n);
	ids.push_back(0);
	expect_error_containing([&] { tree.build_from_ids(ids); }, "too many voxels");
}

TEST_F(VulkanTest, VoxelListAcceptsTheLimitAndRejectsOneMore)
{
	const uint32_t n = VoxelListBase::kMaxEntries;
	std::vector<Coord3ui> coordinates(n);
	for (uint32_t i = 0; i < n; ++i)
		coordinates[i] = cube_voxel(i);

	// one insert of exactly kMaxEntries distinct entries
	CountingVoxelList list(ctx(), { 256, 256, 256 }, 1.0f);
	EXPECT_EQ(list.insert_coordinate_list(coordinates, 1), 0u) << "the return value is the number of dropped coordinates";
	EXPECT_EQ(list.size(), n);

	// the list is full: the merge counts the old and the new entries before it removes duplicates, so any further insert is rejected
	const Coord3ui extra = cube_voxel(n);
	expect_error_containing([&] { (void)list.insert_coordinate_list(std::span<const Coord3ui>(&extra, 1), 1); }, "entries are not supported");
	expect_error_containing([&] { (void)list.insert_coordinate_list(std::span<const Coord3ui>(coordinates.data(), 1), 1); }, "entries are not supported");
	EXPECT_EQ(list.size(), n) << "a rejected insert leaves the list as it was";
	list.clear();

	// kMaxEntries + 1 coordinates (one duplicate) are inserted in chunks of 4 Mi; the merge of the last chunk exceeds the limit
	coordinates.push_back(coordinates.front());
	expect_error_containing([&] { (void)list.insert_coordinate_list(coordinates, 1); }, "entries are not supported");
}

TEST_F(VulkanTest, VoxelListTakesAMetaPointCloudOfExactlyTheLimitOnly)
{
	// a meta point cloud is inserted in one piece (no chunks of 4 Mi): the merge with the list is the only check of its size
	const uint32_t n = VoxelListBase::kMaxEntries;
	CountingVoxelList list(ctx(), { 256, 256, 256 }, 1.0f);
	{
		const uint32_t sizes[] = { n };
		MetaPointCloud cloud(ctx(), sizes);
		EXPECT_NO_THROW((void)list.insert_meta_point_cloud(cloud, 1));
	}
	list.clear();
	{
		const uint32_t sizes[] = { n + 1 };
		MetaPointCloud cloud(ctx(), sizes);
		expect_error_containing([&] { (void)list.insert_meta_point_cloud(cloud, 1); }, "entries are not supported");
	}
}

TEST_F(VulkanTest, ListFromMapIsBoundByTheScan)
{
	// a map of exactly 2^24 voxels is accepted (the scan limit), 2^24 + 1 voxels are not
	{
		BitVoxelMap map(ctx(), { 256, 256, 256 }, 1.0f);
		BitVoxelList list(ctx(), { 256, 256, 256 }, 1.0f);
		EXPECT_EQ(list.insert_occupied_voxels(map), 0u);
	}
	{
		// 97 x 172 961 x 1 = 16 777 217 = 2^24 + 1 voxels (no axis is above 2^24)
		BitVoxelMap map(ctx(), { 97, 172961, 1 }, 1.0f);
		BitVoxelList list(ctx(), { 97, 172961, 1 }, 1.0f);
		expect_error_containing([&] { (void)list.insert_occupied_voxels(map); }, "maps with more than");
	}
}
