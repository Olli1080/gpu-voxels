// Slice S6d-6: Octree::memory_usage() counts the garbage blocks on the GPU (a walk over the reachable blocks, the counts stay in a GPU buffer) instead of
// reading the tree back. The host walk stays as the reference (test hook test_set_host_garbage_count) and as the fallback for a corrupt tree; the tests of
// the insert slices compare the garbage counts with the dynamic reference as well (test_octree_insert.cpp).
#include "vulkan_fixture.h"

#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/OctreeFormat.h>

#include "../reference/octree_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference::octree;
namespace fmt = gpu_voxels::vulkan::octree_format;

namespace
{
	constexpr OctreeKind kKinds[] = { OctreeKind::Deterministic, OctreeKind::Probabilistic };
	const char* name(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? "probabilistic" : "deterministic"; }

	/// The garbage counts of both implementations; they must agree.
	OctreeStats stats_of_both(Octree& tree, const std::string& what, uint64_t* garbage_total = nullptr)
	{
		tree.test_set_host_garbage_count(true);
		const OctreeStats host = tree.memory_usage();
		tree.test_set_host_garbage_count(false);
		const OctreeStats gpu = tree.memory_usage();
		uint64_t total = 0;
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
		{
			EXPECT_EQ(gpu.garbage_blocks[level], host.garbage_blocks[level]) << what << ", level " << level;
			total += host.garbage_blocks[level];
		}
		EXPECT_EQ(gpu.bytes, host.bytes) << what;
		EXPECT_EQ(gpu.chunks, host.chunks) << what;
		if (garbage_total)
			*garbage_total += total;
		return gpu;
	}

	std::vector<OctreeVoxel> cube(uint32_t edge, uint32_t base)
	{
		std::vector<OctreeVoxel> v;
		for (uint32_t x = 0; x < edge; ++x)
			for (uint32_t y = 0; y < edge; ++y)
				for (uint32_t z = 0; z < edge; ++z)
					v.push_back({ base + x, base + y, base + z });
		return v;
	}
}

TEST_F(VulkanTest, OctreeGarbageOnTheGpuEqualsTheHostWalk)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	uint64_t seen_garbage = 0;
	for (const auto kind : kKinds)
	{
		for (uint32_t seed = 0; seed < 8; ++seed)
		{
			SCOPED_TRACE(std::string(name(kind)) + ", seed " + std::to_string(seed));
			std::mt19937_64 rng(seed * 17 + 5);
			Octree tree(ctx(), kind);
			EXPECT_EQ(stats_of_both(tree, "the empty tree").garbage_blocks[0], 0u);

			// uniform cubes (nodes that keep their child blocks), a sparse part far away and, for some seeds, a tree loaded in small chunks
			std::vector<OctreeVoxel> voxels = cube(8, 0);
			for (const auto& v : cube(4, 64))
				voxels.push_back(v);
			for (int i = 0; i < 150; ++i)
				voxels.push_back({ 200 + uint32_t(rng() % 40), uint32_t(rng() % 40), uint32_t(rng() % 40) });
			tree.build(voxels);
			if (seed % 3 == 1)
				tree.load_nodes(tree.dump(), 3);
			stats_of_both(tree, "after the build", &seen_garbage);

			for (int step = 0; step < 10; ++step)
			{
				const std::string what = "step " + std::to_string(step);
				const auto datum = rng() % 2 ? OctreeDatum::Free : OctreeDatum::Occupied;
				const bool propagate = rng() % 4 != 0;
				switch (rng() % 4)
				{
				case 0: // single voxels, some inside the uniform cubes (cut a uniform node: garbage), some new
				{
					std::vector<OctreeVoxel> v;
					for (int i = 0; i < 1 + int(rng() % 6); ++i)
						v.push_back({ uint32_t(rng() % 70), uint32_t(rng() % 10), uint32_t(rng() % 10) });
					tree.insert(v, datum, propagate);
					break;
				}
				case 1: // nodes of a higher level: hard cuts and soft updates
				{
					const uint32_t level = 1 + uint32_t(rng() % 3);
					std::vector<uint64_t> ids;
					for (int i = 0; i < 1 + int(rng() % 4); ++i)
						ids.push_back(ref::morton(uint32_t(rng() % 70), uint32_t(rng() % 10), uint32_t(rng() % 10)) >> (3 * level));
					std::sort(ids.begin(), ids.end());
					ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
					tree.insert_nodes(level, ids, datum, rng() % 2 ? OctreeInsertMode::Soft : OctreeInsertMode::Hard, propagate);
					break;
				}
				case 2: // a batch of voxels spread over the sparse part: many new blocks, several chunks
				{
					std::vector<OctreeVoxel> v;
					for (int i = 0; i < 100; ++i)
						v.push_back({ 200 + uint32_t(rng() % 60), uint32_t(rng() % 60), uint32_t(rng() % 60) });
					tree.insert(v, datum, propagate);
					break;
				}
				default: // a pending insert that is not propagated
					tree.insert(std::vector<OctreeVoxel>{ { uint32_t(rng() % 8), uint32_t(rng() % 8), uint32_t(rng() % 8) } }, datum, false);
					break;
				}
				stats_of_both(tree, what, &seen_garbage);
			}
			EXPECT_EQ(tree.test_garbage_host_fallbacks(), 0u) << "the GPU count gave up on a valid tree";
			tree.rebuild();
			const OctreeStats after = stats_of_both(tree, "after the rebuild");
			for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
				EXPECT_EQ(after.garbage_blocks[level], 0u) << "after the rebuild, level " << level;
		}
	}
	EXPECT_GT(seen_garbage, 0u) << "the sequences never made garbage";
}

// Garbage that can be derived by hand: a free voxel inside a uniform level 2 node of a 4^3 cube leaves its level 1 block and the 8 leaf blocks below
// behind (1 + 8, as in the insert tests); a uniform node of the 8^3 cube cut at level 3 leaves its whole stale subtree.
TEST_F(VulkanTest, OctreeGarbageOfKnownCuts)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		tree.build(cube(4, 0));
		tree.insert(std::vector<OctreeVoxel>{ { 0, 0, 0 } }, OctreeDatum::Free);
		const OctreeStats stats = stats_of_both(tree, "a free voxel in the 4^3 cube");
		EXPECT_EQ(stats.garbage_blocks[1], 1u);
		EXPECT_EQ(stats.garbage_blocks[0], 8u);
		for (uint32_t level = 2; level < 14; ++level)
			EXPECT_EQ(stats.garbage_blocks[level], 0u) << level;
		EXPECT_EQ(tree.test_garbage_host_fallbacks(), 0u);
		tree.rebuild();
		EXPECT_EQ(stats_of_both(tree, "rebuilt").garbage_blocks[0], 0u);
	}
}

// A corrupt tree (a block referenced twice) is not counted by the kernels (more blocks reached than allocated): memory_usage() falls back to the host walk
// and does not throw.
TEST_F(VulkanTest, OctreeGarbageFallsBackToTheHostWalkForACorruptTree)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		tree.build(cube(2, 0));
		tree.insert(std::vector<OctreeVoxel>{ { 0, 0, 0 } }, OctreeDatum::Free); // garbage: sets may_have_garbage
		// a second node of the level 13 block that shares the child block of the first one
		std::vector<uint8_t> block(fmt::inner_block_bytes(kind));
		tree.chunk(13).download(block.data(), block.size());
		fmt::HostNode part{};
		bool found = false;
		for (uint32_t slot = 0; slot < 8 && !found; ++slot)
		{
			part = fmt::decode_inner(kind, block.data(), slot);
			found = (part.status & fmt::kPart) != 0;
		}
		ASSERT_TRUE(found);
		for (uint32_t slot = 0; slot < 8; ++slot) // all 8 nodes share the child block: 8 reachable blocks where 1 was allocated, more than the lists hold
			fmt::encode_inner(kind, block.data(), slot, part);
		tree.chunk(13).upload(block.data(), block.size());

		tree.test_set_host_garbage_count(true);
		OctreeStats host;
		ASSERT_NO_THROW(host = tree.memory_usage());
		tree.test_set_host_garbage_count(false);
		OctreeStats gpu;
		ASSERT_NO_THROW(gpu = tree.memory_usage());
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
			EXPECT_EQ(gpu.garbage_blocks[level], host.garbage_blocks[level]) << level;
		EXPECT_EQ(gpu.bytes, host.bytes);
		EXPECT_EQ(tree.test_garbage_host_fallbacks(), 1u) << "the corrupt tree must be counted on the host";
	}
}

// needs_rebuild() with the garbage fraction takes its numbers from memory_usage(): the same decision as with the host walk, at the border of the fraction.
TEST_F(VulkanTest, OctreeNeedsRebuildDecidesLikeTheHostWalk)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		tree.build(cube(4, 0));
		tree.insert(std::vector<OctreeVoxel>{ { 1, 1, 1 } }, OctreeDatum::Free);
		const OctreeStats stats = tree.memory_usage();
		uint64_t garbage = 0;
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
			garbage += stats.garbage_blocks[level] * fmt::block_bytes(kind, level);
		ASSERT_GT(garbage, 0u);
		const double fraction = double(garbage) / double(stats.bytes);
		for (const bool host : { true, false })
		{
			tree.test_set_host_garbage_count(host);
			tree.set_rebuild_policy({ 0, fraction * 0.99 });
			EXPECT_TRUE(tree.needs_rebuild()) << host;
			tree.set_rebuild_policy({ 0, std::min(1.0, fraction * 1.01) });
			EXPECT_FALSE(tree.needs_rebuild()) << host;
		}
	}
}
