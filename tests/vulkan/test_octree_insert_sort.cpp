// Insert batches above `gpu_sort_limit` ids are sorted and made distinct on the GPU (radix sort, heads, scan, compaction), smaller ones on the
// host. Both paths must give the same tree, the same dropped count and the same cubes for batches with duplicates, unsorted ids and ids outside
// the tree, for single voxels and for nodes of a higher level.
#include "vulkan_fixture.h"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>

#include "../reference/octree_dynamic_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference::octree;

namespace
{
	constexpr OctreeKind kKinds[] = { OctreeKind::Deterministic, OctreeKind::Probabilistic };
	const char* name(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? "probabilistic" : "deterministic"; }

	/// `count` random ids of `level` below `limit` ids, with duplicates (every id appears about twice), unsorted, and about 5 % outside the level.
	std::vector<uint64_t> messy_ids(std::mt19937_64& rng, size_t count, uint32_t level, uint32_t extent)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<uint64_t> ids;
		for (size_t i = 0; i < count; ++i)
			ids.push_back(ref::morton(d(rng), d(rng), d(rng)) >> (3 * level));
		const size_t distinct = ids.size();
		for (size_t i = 0; i < distinct; ++i)
			ids.push_back(ids[rng() % distinct]);
		const uint64_t limit = uint64_t(1) << (3 * (14 - level));
		for (size_t i = 0; i < count / 20 + 1; ++i)
			ids.push_back(limit + rng() % limit);
		std::shuffle(ids.begin(), ids.end(), rng);
		return ids;
	}

	std::vector<OctreeCube> sorted_cubes(const Octree& tree)
	{
		auto cubes = tree.extract();
		std::sort(cubes.begin(), cubes.end(), [](const OctreeCube& a, const OctreeCube& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
		return cubes;
	}
}

TEST_F(VulkanTest, OctreeInsertSortedOnTheGpuEqualsTheHostSort)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		for (uint32_t seed = 0; seed < 4; ++seed)
		{
			SCOPED_TRACE(std::string(name(kind)) + ", seed " + std::to_string(seed));
			std::mt19937_64 rng(seed * 31 + 3);
			Octree gpu(ctx(), kind), host(ctx(), kind);
			gpu.set_gpu_sort_limit(0);          // every batch on the GPU
			host.set_gpu_sort_limit(1u << 30);  // every batch on the host
			std::vector<OctreeVoxel> base;
			for (int i = 0; i < 300; ++i)
				base.push_back({ uint32_t(rng() % 32), uint32_t(rng() % 32), uint32_t(rng() % 32) });
			gpu.build(base);
			host.build(base);
			// batch sizes: a single id, two ids, a few, hundreds, thousands (with the duplicates and invalid ids added on top)
			for (const size_t count : { size_t(1), size_t(2), size_t(7), size_t(500), size_t(5000) })
			{
				const uint32_t level = count == 7 ? 2 : 0;
				const auto ids = messy_ids(rng, count, level, 32);
				const bool free_voxels = count % 2 == 0;
				const auto datum = free_voxels ? OctreeDatum::Free : OctreeDatum::Occupied;
				if (level == 0)
				{
					gpu.insert_ids(ids, datum);
					host.insert_ids(ids, datum);
				}
				else
				{
					gpu.insert_nodes(level, ids, datum, OctreeInsertMode::Hard);
					host.insert_nodes(level, ids, datum, OctreeInsertMode::Hard);
				}
				EXPECT_EQ(gpu.dropped_count(), host.dropped_count()) << "batch of " << count;
				EXPECT_EQ(gpu.dump(), host.dump()) << "batch of " << count;
			}
			// ids over the whole 2^14 grid use all 42 bits of the sort key (and, as nodes of level 3, all 33 bits of the node id)
			for (const uint32_t level : { 0u, 3u })
			{
				const auto ids = messy_ids(rng, 3000, level, 16384);
				if (level == 0)
				{
					gpu.insert_ids(ids);
					host.insert_ids(ids);
				}
				else
				{
					gpu.insert_nodes(level, ids, OctreeDatum::Free, OctreeInsertMode::Hard);
					host.insert_nodes(level, ids, OctreeDatum::Free, OctreeInsertMode::Hard);
				}
				EXPECT_EQ(gpu.dropped_count(), host.dropped_count()) << "wide batch of level " << level;
				EXPECT_EQ(gpu.dump(), host.dump()) << "wide batch of level " << level;
			}
			EXPECT_NO_THROW(gpu.check_tree());
			EXPECT_EQ(sorted_cubes(gpu), sorted_cubes(host));
			EXPECT_GT(gpu.dropped_count(), 0u);
		}
	}
}

TEST_F(VulkanTest, OctreeInsertOfALargeBatchTakesTheGpuSortAndEqualsTheHostSort)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		std::mt19937_64 rng(99);
		Octree gpu(ctx(), kind), host(ctx(), kind);
		host.set_gpu_sort_limit(1u << 30);
		const auto ids = messy_ids(rng, 40000, 0, 64); // 80 000 ids with duplicates: above the default limit of 20 000
		ASSERT_GT(ids.size(), 20000u);
		gpu.insert_ids(ids);
		host.insert_ids(ids);
		EXPECT_EQ(gpu.dropped_count(), host.dropped_count());
		EXPECT_EQ(gpu.dump(), host.dump());
		EXPECT_NO_THROW(gpu.check_tree());
	}
}

// Propagate: the bounded path (one batch, the counts stay on the GPU) and the exact path (counts read back per level) give the same tree.
TEST_F(VulkanTest, OctreePropagateInOneBatchEqualsThePerLevelPropagate)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		for (uint32_t seed = 0; seed < 6; ++seed)
		{
			SCOPED_TRACE(std::string(name(kind)) + ", seed " + std::to_string(seed));
			std::mt19937_64 rng(seed * 17 + 1);
			Octree bounded(ctx(), kind), exact(ctx(), kind);
			exact.set_bounded_propagate_limit(0);
			std::vector<OctreeVoxel> base;
			for (int i = 0; i < 400; ++i)
				base.push_back({ uint32_t(rng() % 40), uint32_t(rng() % 40), uint32_t(rng() % 40) });
			bounded.build(base);
			exact.build(base);
			for (int step = 0; step < 8; ++step)
			{
				// several inserts of both kinds before one propagate: the bound is the sum of the batches
				const int inserts = 1 + int(rng() % 3);
				for (int k = 0; k < inserts; ++k)
				{
					const uint32_t level = rng() % 3 == 0 ? 1 + uint32_t(rng() % 3) : 0;
					const auto ids = messy_ids(rng, 1 + rng() % 60, level, 40);
					const auto datum = rng() % 2 ? OctreeDatum::Occupied : OctreeDatum::Free;
					if (level == 0)
					{
						bounded.insert_ids(ids, datum, false);
						exact.insert_ids(ids, datum, false);
					}
					else
					{
						bounded.insert_nodes(level, ids, datum, OctreeInsertMode::Hard, false);
						exact.insert_nodes(level, ids, datum, OctreeInsertMode::Hard, false);
					}
				}
				bounded.propagate();
				exact.propagate();
				ASSERT_EQ(bounded.dump(), exact.dump()) << "step " << step;
				EXPECT_EQ(bounded.pending_update_count(), 0u);
			}
			EXPECT_NO_THROW(bounded.check_tree());
		}
	}
}

TEST_F(VulkanTest, OctreePropagateRefusesABoundThatIsTooSmall)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		std::mt19937_64 rng(5);
		Octree tree(ctx(), kind);
		tree.insert_ids(messy_ids(rng, 3000, 0, 64), OctreeDatum::Occupied, false);
		tree.test_set_pending_bound(1); // 3000 voxels flagged many nodes on the middle levels
		EXPECT_THROW(tree.propagate(), VulkanError);
	}
}

// Inserting voxels that already exist updates leaves in place: not one block may be allocated for them.
TEST_F(VulkanTest, OctreeInsertOfExistingVoxelsAllocatesNoBlocks)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		std::mt19937_64 rng(21);
		std::vector<OctreeVoxel> voxels;
		for (int i = 0; i < 600; ++i)
			voxels.push_back({ uint32_t(rng() % 40), uint32_t(rng() % 40), uint32_t(rng() % 40) });
		Octree tree(ctx(), kind);
		tree.build(voxels);
		const OctreeStats before = tree.memory_usage();
		for (const auto datum : { OctreeDatum::Occupied, OctreeDatum::Free })
		{
			tree.insert(voxels, datum);
			const OctreeStats after = tree.memory_usage();
			EXPECT_EQ(after.bytes, before.bytes);
			EXPECT_EQ(after.chunks, before.chunks);
			for (uint32_t level = 0; level < 15; ++level)
				EXPECT_EQ(after.blocks[level], before.blocks[level]) << "level " << level;
			EXPECT_NO_THROW(tree.check_tree());
		}
	}
}

// More pending ids than the bounded path takes (default limit 131 072): propagate runs the exact path. Its work lists of the big levels are sized
// exactly after a count pass (threshold 4 MiB; here 150 000 ids make lists of 1 - 2 MB, so the threshold is lowered to 0 for one tree and raised to
// never for another). All of them must equal the bounded path with the limit raised (the same kernels, lists sized from the bound), for single voxels
// and for a soft insert of nodes (which makes the bound unknown anyway).
TEST_F(VulkanTest, OctreeExactPropagateOfManyPendingIdsEqualsTheBoundedPath)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		for (uint32_t seed = 0; seed < 2; ++seed)
		{
			SCOPED_TRACE(std::string(name(kind)) + ", seed " + std::to_string(seed));
			std::mt19937_64 rng(seed + 1000);
			Octree exact(ctx(), kind), counted(ctx(), kind), plain(ctx(), kind), bounded(ctx(), kind);
			bounded.set_bounded_propagate_limit(uint64_t(1) << 40);
			counted.set_exact_list_threshold(0);
			plain.set_exact_list_threshold(UINT64_MAX);
			Octree* const trees[] = { &exact, &counted, &plain, &bounded };
			std::vector<OctreeVoxel> base;
			for (int i = 0; i < 2000; ++i)
				base.push_back({ uint32_t(rng() % 64), uint32_t(rng() % 64), uint32_t(rng() % 64) });
			for (Octree* tree : trees)
				tree->build(base);
			for (int round = 0; round < 3; ++round)
			{
				std::vector<uint64_t> ids;
				for (int i = 0; i < 150000; ++i)
					ids.push_back(ref::morton(uint32_t(rng() % 64), uint32_t(rng() % 64), uint32_t(rng() % 64)));
				const auto datum = round == 1 ? OctreeDatum::Free : OctreeDatum::Occupied;
				for (Octree* tree : trees)
					tree->insert_ids(ids, datum, false);
				if (round == 2) // a soft insert on top: the exact path for all
				{
					const std::vector<uint64_t> nodes = { 3, 5, 9 };
					for (Octree* tree : trees)
						tree->insert_nodes(2, nodes, OctreeDatum::Free, OctreeInsertMode::Soft, false);
				}
				for (Octree* tree : trees)
					tree->propagate();
				EXPECT_EQ(exact.pending_update_count(), 0u);
				EXPECT_EQ(exact.dump(), bounded.dump()) << "round " << round;
				EXPECT_EQ(counted.dump(), bounded.dump()) << "counted, round " << round;
				EXPECT_EQ(plain.dump(), bounded.dump()) << "plain, round " << round;
				EXPECT_NO_THROW(exact.check_tree());
			}
		}
	}
}

// The exact path on random sequences with soft node inserts (subtree updates reach whole subtrees, so the lists are not bounded by the inserts): lists
// sized exactly after a count pass on every level (threshold 0) give the same tree as lists of 8 entries per node (threshold never) and as the
// per-level reference schedule the earlier tests compare against.
TEST_F(VulkanTest, OctreeExactPropagateWithCountedListsEqualsUncountedLists)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		for (uint32_t seed = 0; seed < 6; ++seed)
		{
			SCOPED_TRACE(std::string(name(kind)) + ", seed " + std::to_string(seed));
			std::mt19937_64 rng(seed * 13 + 7);
			Octree counted(ctx(), kind), plain(ctx(), kind);
			for (Octree* tree : { &counted, &plain })
				tree->set_bounded_propagate_limit(0); // always the exact path
			counted.set_exact_list_threshold(0);
			plain.set_exact_list_threshold(UINT64_MAX);
			std::vector<OctreeVoxel> base;
			for (int i = 0; i < 400; ++i)
				base.push_back({ uint32_t(rng() % 32), uint32_t(rng() % 32), uint32_t(rng() % 32) });
			counted.build(base);
			plain.build(base);
			for (int step = 0; step < 8; ++step)
			{
				const bool propagate = rng() % 3 != 0;
				const auto datum = rng() % 2 ? OctreeDatum::Free : OctreeDatum::Occupied;
				if (rng() % 2)
				{
					const auto ids = messy_ids(rng, 200, 0, 32);
					counted.insert_ids(ids, datum, propagate);
					plain.insert_ids(ids, datum, propagate);
				}
				else
				{
					const uint32_t level = 1 + uint32_t(rng() % 3);
					const auto ids = messy_ids(rng, 4, level, 32 >> level);
					const auto mode = rng() % 2 ? OctreeInsertMode::Soft : OctreeInsertMode::Hard;
					counted.insert_nodes(level, ids, datum, mode, propagate);
					plain.insert_nodes(level, ids, datum, mode, propagate);
				}
				EXPECT_EQ(counted.dump(), plain.dump()) << "step " << step;
			}
			counted.propagate();
			plain.propagate();
			EXPECT_EQ(counted.dump(), plain.dump());
			EXPECT_NO_THROW(counted.check_tree());
		}
	}
}
