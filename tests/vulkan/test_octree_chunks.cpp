// Slice S6d-0: a level of the octree is a list of chunks. A tree spread over several chunks per level (load_nodes) must behave exactly
// like the tree with one chunk per level: dump, check_tree, find, extract (both implementations), collisions, collision flags.
#include "vulkan_fixture.h"

#include <algorithm>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>

#include "../reference/octree_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference::octree;
namespace fmt = gpu_voxels::vulkan::octree_format;

namespace
{
	constexpr OctreeKind kKinds[] = { OctreeKind::Deterministic, OctreeKind::Probabilistic };
	const char* name(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? "probabilistic" : "deterministic"; }

	std::vector<OctreeVoxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent, uint32_t ox = 0, uint32_t oy = 0, uint32_t oz = 0)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<OctreeVoxel> v;
		for (size_t i = 0; i < count; ++i)
			v.push_back({ ox + d(rng), oy + d(rng), oz + d(rng) });
		return v;
	}

	std::vector<OctreeVoxel> cube(uint32_t x0, uint32_t y0, uint32_t z0, uint32_t side)
	{
		std::vector<OctreeVoxel> v;
		for (uint32_t z = 0; z < side; ++z)
			for (uint32_t y = 0; y < side; ++y)
				for (uint32_t x = 0; x < side; ++x)
					v.push_back({ x0 + x, y0 + y, z0 + z });
		return v;
	}

	std::vector<OctreeCube> sorted(std::vector<OctreeCube> cubes)
	{
		std::sort(cubes.begin(), cubes.end(), [](const OctreeCube& a, const OctreeCube& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
		return cubes;
	}

	uint32_t max_chunks(const Octree& tree)
	{
		uint32_t most = 0;
		for (uint32_t level = 0; level < Octree::kLevels; ++level)
			most = std::max(most, tree.chunk_count(level));
		return most;
	}

	std::vector<uint64_t> ids_of(const std::vector<OctreeVoxel>& voxels)
	{
		std::vector<uint64_t> ids;
		for (const auto& v : voxels)
			ids.push_back(ref::morton(v.x, v.y, v.z));
		return ids;
	}
}

TEST_F(VulkanTest, OctreeStartsWithOneChunkPerLevelAndNoGarbage)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(101);
	for (const auto kind : kKinds)
	{
		Octree tree(ctx(), kind);
		EXPECT_EQ(tree.chunk_count(fmt::kRootLevel), 1u) << name(kind);
		EXPECT_EQ(tree.chunk_count(3), 0u);
		EXPECT_EQ(tree.chunk_count(99), 0u);
		tree.build(random_voxels(rng, 700, 40));
		const auto stats = tree.memory_usage();
		uint32_t levels_with_blocks = 0;
		for (uint32_t level = 0; level < Octree::kLevels; ++level)
		{
			EXPECT_EQ(tree.chunk_count(level), stats.blocks[level] > 0 ? 1u : 0u) << "level " << level;
			EXPECT_EQ(stats.garbage_blocks[level], 0u);
			levels_with_blocks += stats.blocks[level] > 0;
		}
		EXPECT_EQ(stats.chunks, levels_with_blocks);
		EXPECT_THROW((void)tree.chunk(2, 1), VulkanError);
		EXPECT_THROW((void)tree.chunk(Octree::kLevels, 0), VulkanError);
		EXPECT_NO_THROW((void)tree.chunk(0, 0));
	}
}

TEST_F(VulkanTest, LoadNodesReproducesATreeInOneChunkPerLevel)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(102);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree original(ctx(), kind), copy(ctx(), kind);
		original.build(random_voxels(rng, 1500, 50, 300, 400, 500));
		copy.load_nodes(original.dump());
		EXPECT_EQ(copy.dump(), original.dump());
		EXPECT_NO_THROW(copy.check_tree());
		const auto a = original.memory_usage(), b = copy.memory_usage();
		for (uint32_t level = 0; level < Octree::kLevels; ++level)
			EXPECT_EQ(a.blocks[level], b.blocks[level]) << "level " << level;
		EXPECT_EQ(copy.voxel_count(), 0u);
	}
}

TEST_F(VulkanTest, TreeInSeveralChunksPerLevelBehavesLikeOneChunkPerLevel)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(103);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		auto voxels = random_voxels(rng, 1200, 36, 100, 100, 100);
		const auto block = cube(200, 200, 200, 8);
		voxels.insert(voxels.end(), block.begin(), block.end());
		voxels.push_back({ 16383, 16383, 16383 });
		voxels.push_back({ 0, 0, 0 });
		const auto probes = random_voxels(rng, 1800, 40, 95, 95, 95);

		Octree whole(ctx(), kind), split(ctx(), kind), other(ctx(), kind);
		whole.build(voxels);
		other.build(random_voxels(rng, 1000, 36, 110, 110, 110));
		for (const uint64_t chunk_blocks : { 1ull, 3ull, 7ull, 50ull })
		{
			SCOPED_TRACE("blocks per chunk " + std::to_string(chunk_blocks));
			split.load_nodes(whole.dump(), chunk_blocks);
			ASSERT_GT(max_chunks(split), 1u) << "the test needs more than one chunk on some level";
			EXPECT_EQ(split.dump(), whole.dump());
			ASSERT_NO_THROW(split.check_tree());
			const auto stats = split.memory_usage();
			EXPECT_GT(stats.chunks, 15u);
			for (uint32_t level = 0; level < Octree::kLevels; ++level)
				EXPECT_LE(split.chunk_count(level), (stats.blocks[level] + chunk_blocks - 1) / chunk_blocks);

			// find
			std::vector<uint64_t> ids = ids_of(probes);
			ids.push_back(1ull << 41);
			const auto f_whole = whole.find(ids), f_split = split.find(ids);
			ASSERT_EQ(f_whole.size(), f_split.size());
			for (size_t i = 0; i < ids.size(); ++i)
				ASSERT_TRUE(f_whole[i].level == f_split[i].level && f_whole[i].status == f_split[i].status && f_whole[i].occupancy == f_split[i].occupancy) << "id " << ids[i];

			// extract, wavefront and load balancer
			for (const bool balancer : { false, true })
				for (const uint32_t min_level : { 0u, 2u, 5u })
				{
					OctreeExtractOptions options;
					options.min_level = min_level;
					options.use_load_balancer = balancer;
					ASSERT_TRUE(sorted(whole.extract(options)) == sorted(split.extract(options))) << "balancer " << balancer << " min_level " << min_level;
				}

			// tree vs tree in all four combinations of split / whole, with marks; both trees keep identical dumps
			for (const bool balancer : { false, true })
			{
				OctreeTreeCollisionOptions options;
				options.mark_collisions = true;
				options.use_load_balancer = balancer;
				const uint64_t expected = whole.collide_with(other, options);
				const auto expected_whole = whole.dump(), expected_other = other.dump();
				whole.clear_collision_flags();
				other.clear_collision_flags();
				const uint64_t volume = split.collide_with(other, options);
				EXPECT_EQ(volume, expected) << "balancer " << balancer;
				EXPECT_TRUE(split.dump() == expected_whole) << "marked nodes of the split tree, balancer " << balancer;
				EXPECT_TRUE(other.dump() == expected_other) << "marked nodes of the other tree, balancer " << balancer;
				split.clear_collision_flags();
				other.clear_collision_flags();
				EXPECT_TRUE(split.dump() == whole.dump()) << "clear_collision_flags reaches every chunk";
				EXPECT_EQ(other.collide_with(split, options), expected) << "the split tree as the other operand, balancer " << balancer;
				other.clear_collision_flags();
				split.clear_collision_flags();
			}

			// probes against the tree
			OctreeCollisionOptions probe_options;
			const auto c_whole = whole.collide_with_ids(ids_of(probes), {}, probe_options);
			const auto c_split = split.collide_with_ids(ids_of(probes), {}, probe_options);
			EXPECT_EQ(c_whole.collisions, c_split.collisions);
			whole.clear_collision_flags();
			split.clear_collision_flags();
		}
	}
}

TEST_F(VulkanTest, LoadNodesRejectsInconsistentLists)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(104);
	Octree tree(ctx(), OctreeKind::Deterministic);
	tree.build(random_voxels(rng, 300, 20, 50, 50, 50));
	const auto good = tree.dump();
	const auto before = tree.dump();

	auto without = [&](auto predicate)
	{
		std::vector<OctreeNode> v;
		for (const auto& n : good)
			if (!predicate(n))
				v.push_back(n);
		return v;
	};
	const uint32_t level_with_nodes = 3;
	const OctreeNode victim = *std::find_if(good.begin(), good.end(), [&](const OctreeNode& n) { return n.level == level_with_nodes; });

	EXPECT_THROW(tree.load_nodes(without([](const OctreeNode& n) { return n.level == fmt::kRootLevel; })), VulkanError) << "no root";
	EXPECT_THROW(tree.load_nodes(without([&](const OctreeNode& n) { return n.level == victim.level && n.id == victim.id; })), VulkanError) << "an incomplete block";
	EXPECT_THROW(tree.load_nodes(without([](const OctreeNode& n) { return n.level == 5; })), VulkanError) << "children without their block";
	{
		auto twice = good;
		twice.push_back(victim);
		EXPECT_THROW(tree.load_nodes(twice), VulkanError) << "a node listed twice";
	}
	{
		auto leaf_with_children = good;
		for (auto& n : leaf_with_children)
			if (n.level == 0)
			{
				n.has_children = true;
				break;
			}
		EXPECT_THROW(tree.load_nodes(leaf_with_children), VulkanError) << "a leaf with children";
	}
	{
		auto orphan = good;
		for (auto& n : orphan)
			if (n.level == 6 && n.has_children)
			{
				n.has_children = false;
				break;
			}
		EXPECT_THROW(tree.load_nodes(orphan), VulkanError) << "a block without a parent node";
	}
	{
		auto bad_id = good;
		bad_id.push_back({ 14, 5, 0, 0, false });
		EXPECT_THROW(tree.load_nodes(bad_id), VulkanError) << "a second root";
		auto out_of_range = good;
		out_of_range.push_back({ 13, 1ull << 20, 0, 0, false });
		EXPECT_THROW(tree.load_nodes(out_of_range), VulkanError) << "an id outside the level";
		auto bad_level = good;
		bad_level.push_back({ 15, 0, 0, 0, false });
		EXPECT_THROW(tree.load_nodes(bad_level), VulkanError) << "a level above the root";
	}
	EXPECT_TRUE(tree.dump() == before) << "a rejected list leaves the old tree";
	EXPECT_NO_THROW(tree.check_tree());

	// the empty tree and a tree that is only a root round trip
	Octree empty(ctx(), OctreeKind::Probabilistic), copy(ctx(), OctreeKind::Probabilistic);
	copy.build(random_voxels(rng, 100, 10));
	copy.load_nodes(empty.dump());
	EXPECT_EQ(copy.dump(), empty.dump());
	EXPECT_NO_THROW(copy.check_tree());
	EXPECT_EQ(copy.memory_usage().chunks, 1u);
}
