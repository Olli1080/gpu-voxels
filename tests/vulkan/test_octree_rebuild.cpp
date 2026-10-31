// Slice S6d-4: Octree::rebuild() and needs_rebuild(). Rebuild compacts the tree on the GPU: afterwards every node reachable through PART nodes is
// unchanged, nothing else is allocated (one chunk per level, no garbage, no stale blocks below nodes without PART). The expected node list is
// computed on the host from dump() by a plain walk over PART nodes (independent of the kernels); extract and find must not notice the rebuild.
#include "vulkan_fixture.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/OctreeFormat.h>

#include "../reference/octree_dynamic_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference::octree;
namespace fmt = gpu_voxels::vulkan::octree_format;

namespace
{
	constexpr OctreeKind kKinds[] = { OctreeKind::Deterministic, OctreeKind::Probabilistic };
	const char* name(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? "probabilistic" : "deterministic"; }

	std::vector<OctreeVoxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent, uint32_t base)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<OctreeVoxel> v;
		for (size_t i = 0; i < count; ++i)
			v.push_back({ base + d(rng), base + d(rng), base + d(rng) });
		return v;
	}

	std::vector<uint64_t> node_ids_of(const std::vector<OctreeVoxel>& voxels, uint32_t level)
	{
		std::set<uint64_t> ids;
		for (const auto& v : voxels)
			ids.insert(ref::morton(v.x, v.y, v.z) >> (3 * level));
		return { ids.begin(), ids.end() };
	}

	struct Op
	{
		uint32_t level;
		std::vector<uint64_t> ids;
		OctreeDatum datum;
		OctreeInsertMode mode;
		bool propagate;
	};

	/// Random inserts into the cube of 16 voxels: single voxels and nodes of level 1 .. 4, hard and soft, propagated now or later.
	std::vector<Op> random_ops(std::mt19937_64& rng, size_t steps, bool last_propagates)
	{
		std::vector<Op> ops;
		for (size_t i = 0; i < steps; ++i)
		{
			const uint32_t level = rng() % 10 < 5 ? 0 : 1 + static_cast<uint32_t>(rng() % 4);
			const size_t count = level == 0 ? 20 + rng() % 60 : 1 + rng() % 6;
			ops.push_back({ level, node_ids_of(random_voxels(rng, count, 16, 0), level), rng() % 2 ? OctreeDatum::Occupied : OctreeDatum::Free,
				rng() % 2 ? OctreeInsertMode::Soft : OctreeInsertMode::Hard, rng() % 3 != 0 });
		}
		ops.back().propagate = last_propagates;
		return ops;
	}

	void apply(Octree& tree, const std::vector<Op>& ops)
	{
		for (const Op& op : ops)
		{
			if (op.level == 0)
				tree.insert_ids(op.ids, op.datum, op.propagate);
			else
				tree.insert_nodes(op.level, op.ids, op.datum, op.mode, op.propagate);
		}
	}

	/// The nodes of dump() reachable through PART nodes, as a rebuilt tree has to list them: a node without PART has no child block.
	std::vector<OctreeNode> live_nodes(const std::vector<OctreeNode>& dump)
	{
		std::map<std::pair<uint32_t, uint64_t>, const OctreeNode*> by_key;
		for (const auto& n : dump)
			by_key[{ n.level, n.id }] = &n;
		std::vector<OctreeNode> out;
		std::vector<const OctreeNode*> stack{ by_key.at({ 14, 0 }) };
		while (!stack.empty())
		{
			const OctreeNode* n = stack.back();
			stack.pop_back();
			OctreeNode copy = *n;
			const bool part = (n->status & fmt::kPart) != 0;
			copy.has_children = part;
			out.push_back(copy);
			if (part)
				for (uint64_t k = 0; k < 8; ++k)
					stack.push_back(by_key.at({ n->level - 1, n->id * 8 + k }));
		}
		std::sort(out.begin(), out.end(), [](const OctreeNode& a, const OctreeNode& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
		return out;
	}

	std::vector<OctreeCube> sorted_cubes(const Octree& tree)
	{
		auto cubes = tree.extract();
		std::sort(cubes.begin(), cubes.end(), [](const OctreeCube& a, const OctreeCube& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
		return cubes;
	}

	std::string describe(const OctreeNode& n)
	{
		return "(level " + std::to_string(n.level) + ", id " + std::to_string(n.id) + ", status " + std::to_string(n.status) + ", occupancy " +
			std::to_string(n.occupancy) + (n.has_children ? ", children)" : ")");
	}

	std::string difference(const std::vector<OctreeNode>& got, const std::vector<OctreeNode>& expected)
	{
		if (got.size() != expected.size())
			return "node counts differ: " + std::to_string(got.size()) + " instead of " + std::to_string(expected.size());
		for (size_t i = 0; i < got.size(); ++i)
			if (!(got[i] == expected[i]))
				return "node " + std::to_string(i) + ": " + describe(got[i]) + " instead of " + describe(expected[i]);
		return {};
	}

	/// The tree must list exactly `expected`, own one chunk per level that has blocks and no garbage.
	void expect_compact(const Octree& tree, const std::vector<OctreeNode>& expected, const std::string& what)
	{
		EXPECT_EQ(difference(tree.dump(), expected), "") << what;
		EXPECT_NO_THROW(tree.check_tree()) << what;
		EXPECT_EQ(tree.pending_update_count(), 0u) << what;
		const OctreeStats stats = tree.memory_usage();
		uint64_t nodes_of_level[fmt::kLevelCount] = {};
		for (const auto& n : expected)
			++nodes_of_level[n.level];
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
		{
			const uint64_t blocks = level == 14 ? (nodes_of_level[level] > 0 ? 1 : 0) : nodes_of_level[level] / 8;
			EXPECT_EQ(stats.blocks[level], blocks) << what << ", blocks of level " << level;
			EXPECT_EQ(stats.garbage_blocks[level], 0u) << what << ", garbage of level " << level;
			EXPECT_EQ(tree.chunk_count(level), blocks > 0 ? 1u : 0u) << what << ", chunks of level " << level;
		}
	}

	std::vector<uint64_t> probe_ids(std::mt19937_64& rng, size_t count)
	{
		std::vector<uint64_t> ids;
		for (size_t i = 0; i < count; ++i)
			ids.push_back(i % 2 == 0 ? rng() % 4096 : rng() % (uint64_t(1) << 42));
		return ids;
	}
}

TEST_F(VulkanTest, OctreeRebuildReleasesGarbageAndKeepsEveryLiveNode)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	uint64_t garbage_seen = 0, bytes_saved = 0;
	for (const auto kind : kKinds)
	{
		for (uint32_t seed = 0; seed < 10; ++seed)
		{
			SCOPED_TRACE(std::string(name(kind)) + ", seed " + std::to_string(seed));
			std::mt19937_64 rng(seed * 7919 + 1);
			const auto voxels = random_voxels(rng, 150, 16, 0);
			const auto ops = random_ops(rng, 8, seed % 2 == 0); // odd seeds end with an insert that is not propagated
			Octree plain(ctx(), kind), rebuilt(ctx(), kind);
			plain.build(voxels);
			rebuilt.build(voxels);
			apply(plain, ops);
			apply(rebuilt, ops);
			if (seed % 3 == 0)
				rebuilt.set_max_groups_x(2); // the 2D dispatch path

			plain.propagate();
			const auto expected = live_nodes(plain.dump());
			const OctreeStats before = plain.memory_usage();
			for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
				garbage_seen += before.garbage_blocks[level];
			const auto cubes = sorted_cubes(plain);
			const auto ids = probe_ids(rng, 3000);
			const auto found = plain.find(ids);

			rebuilt.rebuild();
			expect_compact(rebuilt, expected, "after rebuild");
			EXPECT_EQ(sorted_cubes(rebuilt), cubes);
			EXPECT_EQ(rebuilt.find(ids), found);
			EXPECT_EQ(rebuilt.voxel_count(), plain.voxel_count());
			const OctreeStats after = rebuilt.memory_usage();
			EXPECT_LE(after.bytes, before.bytes);
			bytes_saved += before.bytes - after.bytes;

			// a second rebuild changes nothing
			rebuilt.rebuild();
			expect_compact(rebuilt, expected, "after the second rebuild");
		}
	}
	// the scenarios must really contain garbage, otherwise the test proves little
	EXPECT_GT(garbage_seen, 0u);
	EXPECT_GT(bytes_saved, 0u);
}

TEST_F(VulkanTest, OctreeRebuildDropsTheStaleSubtreeBelowACutNode)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		tree.build(std::vector<OctreeVoxel>{ { 0, 0, 0 } });
		tree.insert_nodes(13, std::vector<uint64_t>{ 0 }, OctreeDatum::Free);
		EXPECT_NO_THROW(tree.check_tree());
		// the cut node still points to its old chain: one block on every level
		const OctreeStats before = tree.memory_usage();
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
			EXPECT_EQ(before.blocks[level], 1u) << level;
		const auto cubes = sorted_cubes(tree);

		tree.rebuild();
		const OctreeStats after = tree.memory_usage();
		for (uint32_t level = 0; level < 12; ++level)
			EXPECT_EQ(after.blocks[level], 0u) << level;
		EXPECT_EQ(after.blocks[13], 1u);
		EXPECT_EQ(after.blocks[14], 1u);
		EXPECT_LT(after.bytes, before.bytes);
		const auto dump = tree.dump();
		ASSERT_EQ(dump.size(), 9u);
		for (const auto& n : dump) // the root keeps its block of level 13, the nodes of that block are cut
			EXPECT_EQ(n.has_children, n.level == 14) << describe(n);
		EXPECT_EQ(sorted_cubes(tree), cubes);
		EXPECT_NO_THROW(tree.check_tree());
	}
}

TEST_F(VulkanTest, OctreeRebuildOfTheEmptyAndTheSingleVoxelTree)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree empty(ctx(), kind);
		const auto before = empty.dump();
		empty.rebuild();
		EXPECT_EQ(difference(empty.dump(), before), "");
		EXPECT_EQ(empty.memory_usage().nodes(), 8u);
		// the rebuilt empty tree takes inserts like a fresh one
		const std::vector<OctreeVoxel> voxels = { { 5, 6, 7 }, { 5, 6, 8 } };
		Octree fresh(ctx(), kind);
		empty.insert(voxels);
		fresh.insert(voxels);
		EXPECT_EQ(difference(empty.dump(), fresh.dump()), "");

		Octree one(ctx(), kind);
		one.build(std::vector<OctreeVoxel>{ { 9, 9, 9 } });
		const auto dump = one.dump();
		one.rebuild();
		EXPECT_EQ(difference(one.dump(), dump), "");
		EXPECT_EQ(one.voxel_count(), 1u);
	}
}

TEST_F(VulkanTest, OctreeRebuildJoinsTheChunksOfATreeLoadedInPieces)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(77);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto voxels = random_voxels(rng, 400, 30, 600);
		Octree whole(ctx(), kind), pieces(ctx(), kind);
		whole.build(voxels);
		const auto dump = whole.dump();
		pieces.load_nodes(dump, 3);
		uint32_t most = 0;
		for (uint32_t level = 0; level < 14; ++level)
			most = std::max(most, pieces.chunk_count(level));
		ASSERT_GT(most, 3u);

		pieces.rebuild();
		expect_compact(pieces, live_nodes(dump), "pieces after rebuild");
		EXPECT_EQ(difference(pieces.dump(), whole.dump()), "");

		// the same answers to queries
		std::vector<uint64_t> ids;
		for (const auto& v : random_voxels(rng, 500, 40, 590))
			ids.push_back(ref::morton(v.x, v.y, v.z));
		OctreeCollisionOptions options;
		options.set_collision_flags = false;
		EXPECT_EQ(pieces.collide_with_ids(ids, {}, options), whole.collide_with_ids(ids, {}, options));
		EXPECT_EQ(pieces.find(ids), whole.find(ids));
	}
}

TEST_F(VulkanTest, OctreeRebuildKeepsCollisionBitsAndOccupancies)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(5);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto voxels = random_voxels(rng, 200, 16, 0);
		Octree tree(ctx(), kind);
		tree.build(voxels);
		tree.insert_nodes(2, std::vector<uint64_t>{ 3 }, OctreeDatum::Free);
		std::vector<uint64_t> ids;
		for (const auto& v : random_voxels(rng, 100, 16, 0))
			ids.push_back(ref::morton(v.x, v.y, v.z));
		ASSERT_GT(tree.collide_with_ids(ids).collisions, 0u);
		// probes that stop on level 2 mark inner nodes (an exact probe only marks the voxels)
		OctreeCollisionOptions coarse;
		coarse.min_level = 2;
		ASSERT_GT(tree.collide_with_ids(ids, {}, coarse).collisions, 0u);
		const auto expected = live_nodes(tree.dump());
		size_t marked = 0, marked_inner = 0, with_occupancy = 0;
		for (const auto& n : expected)
		{
			marked += (n.status & fmt::kCollision) != 0;
			marked_inner += n.level > 0 && (n.status & fmt::kCollision) != 0;
			with_occupancy += n.occupancy != 0;
		}
		ASSERT_GT(marked, 0u) << "the collision query must have marked nodes";
		ASSERT_GT(marked_inner, 0u) << "the coarse query must have marked inner nodes";
		if (kind == OctreeKind::Probabilistic)
			ASSERT_GT(with_occupancy, 0u);

		tree.rebuild();
		expect_compact(tree, expected, "after rebuild");
		tree.clear_collision_flags();
		EXPECT_EQ(difference(tree.dump(), [&] { auto cleared = expected; for (auto& n : cleared) n.status &= uint8_t(~fmt::kCollision); return cleared; }()), "");
	}
}

// The 32 bit halves of the addresses in the work lists and the pointers are only exercised when a chunk lies above 4 GiB. A device that hands
// out addresses in ascending order gets there behind some ballast buffers; the test skips where that does not work.
TEST_F(VulkanTest, OctreeRebuildWithChunksAbove4GiB)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	// the allocator hands out small buffers from blocks of 256 MB that follow each other: fill them until the next one lies above 4 GiB
	std::vector<DeviceBuffer> ballast;
	bool high = false;
	for (int i = 0; i < 8000 && !high; ++i)
	{
		ballast.emplace_back(ctx(), uint64_t(1) << 20);
		high = ballast.back().device_address() >= (uint64_t(1) << 32);
	}
	if (!high)
		GTEST_SKIP() << "no buffer above 4 GiB after " << ballast.size() << " ballast buffers";
	std::mt19937_64 rng(11);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto voxels = random_voxels(rng, 150, 16, 0);
		Octree tree(ctx(), kind);
		tree.build(voxels);
		const auto ops = random_ops(rng, 5, true);
		apply(tree, ops);
		bool above = true;
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
			for (uint32_t c = 0; c < tree.chunk_count(level); ++c)
				above = above && tree.chunk(level, c).device_address() >= (uint64_t(1) << 32);
		if (!above)
			GTEST_SKIP() << "the tree was not placed above 4 GiB";
		const auto expected = live_nodes(tree.dump());
		const auto cubes = sorted_cubes(tree);
		tree.rebuild();
		expect_compact(tree, expected, "after rebuild");
		EXPECT_EQ(sorted_cubes(tree), cubes);
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
			for (uint32_t c = 0; c < tree.chunk_count(level); ++c)
				EXPECT_GE(tree.chunk(level, c).device_address(), uint64_t(1) << 32) << "level " << level << ": the rebuilt chunk moved below 4 GiB";
	}
}

TEST_F(VulkanTest, OctreeInsertsAfterARebuildGiveTheTreeWithoutARebuild)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		for (uint32_t seed = 0; seed < 6; ++seed)
		{
			SCOPED_TRACE(std::string(name(kind)) + ", seed " + std::to_string(seed));
			std::mt19937_64 rng(seed * 104729 + 5);
			const auto voxels = random_voxels(rng, 120, 16, 0);
			const auto first = random_ops(rng, 6, true);
			const auto second = random_ops(rng, 6, true);
			Octree control(ctx(), kind), rebuilt(ctx(), kind);
			control.build(voxels);
			rebuilt.build(voxels);
			apply(control, first);
			apply(rebuilt, first);
			rebuilt.rebuild();
			apply(control, second);
			apply(rebuilt, second);
			EXPECT_NO_THROW(control.check_tree());
			EXPECT_NO_THROW(rebuilt.check_tree());
			EXPECT_EQ(difference(live_nodes(rebuilt.dump()), live_nodes(control.dump())), "");
			EXPECT_EQ(sorted_cubes(rebuilt), sorted_cubes(control));
			// and the rebuilt tree can be rebuilt again
			rebuilt.rebuild();
			expect_compact(rebuilt, live_nodes(control.dump()), "rebuilt twice");
		}
	}
}

TEST_F(VulkanTest, OctreeNeedsRebuildFollowsTheMemoryLimitAndTheGarbageFraction)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		const OctreeRebuildPolicy defaults = tree.rebuild_policy();
		EXPECT_EQ(defaults.max_bytes, uint64_t(200) * 1024 * 1024); // CUDA: 200 MB
		EXPECT_EQ(defaults.max_garbage_fraction, 0.0);

		// a full block of 2^3 voxels is a uniform node; a free voxel inside it creates a new block and leaves the old one as garbage
		std::vector<OctreeVoxel> block;
		for (uint32_t i = 0; i < 8; ++i)
			block.push_back({ i & 1, (i >> 1) & 1, (i >> 2) & 1 });
		tree.build(block);
		EXPECT_FALSE(tree.needs_rebuild());
		const uint64_t bytes = tree.memory_usage().bytes;

		tree.set_rebuild_policy({ bytes, 0.0 });
		EXPECT_TRUE(tree.needs_rebuild()) << "the limit counts as reached at exactly the memory in use";
		tree.set_rebuild_policy({ bytes + 1, 0.0 });
		EXPECT_FALSE(tree.needs_rebuild());
		tree.set_rebuild_policy({ 1, 0.0 });
		EXPECT_TRUE(tree.needs_rebuild());
		tree.set_rebuild_policy({ 0, 0.0 });
		EXPECT_FALSE(tree.needs_rebuild()) << "0 = no limit";
		tree.set_rebuild_policy({ 0, 0.5 });
		EXPECT_FALSE(tree.needs_rebuild()) << "no garbage yet";

		tree.insert(std::vector<OctreeVoxel>{ { 0, 0, 0 } }, OctreeDatum::Free);
		const OctreeStats stats = tree.memory_usage();
		uint64_t garbage = 0;
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
			garbage += stats.garbage_blocks[level] * fmt::block_bytes(kind, level);
		ASSERT_GT(garbage, 0u);
		const double fraction = double(garbage) / double(stats.bytes);
		tree.set_rebuild_policy({ 0, fraction * 0.5 });
		EXPECT_TRUE(tree.needs_rebuild());
		tree.set_rebuild_policy({ 0, fraction * 2.0 });
		EXPECT_FALSE(tree.needs_rebuild());
		tree.set_rebuild_policy({ 0, 0.0 });
		EXPECT_FALSE(tree.needs_rebuild()) << "the fraction is off";

		tree.set_rebuild_policy({ 0, fraction * 0.5 });
		tree.rebuild();
		EXPECT_FALSE(tree.needs_rebuild()) << "the garbage is gone";
		EXPECT_EQ(tree.memory_usage().garbage_blocks[0], 0u);
		EXPECT_EQ(tree.rebuild_policy().max_garbage_fraction, fraction * 0.5);

		for (const double invalid : { -0.1, 1.5, std::nan("") })
			EXPECT_THROW(tree.set_rebuild_policy({ 0, invalid }), VulkanError) << invalid;
		EXPECT_EQ(tree.rebuild_policy().max_garbage_fraction, fraction * 0.5) << "a rejected policy changes nothing";
	}
}

TEST_F(VulkanTest, OctreeRebuildRefusesACorruptTreeAndLeavesItUntouched)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		tree.build(std::vector<OctreeVoxel>{ { 1, 2, 3 } });
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
		for (uint32_t slot = 0; slot < 8; ++slot)
			if (!(fmt::decode_inner(kind, block.data(), slot).status & fmt::kPart))
			{
				fmt::encode_inner(kind, block.data(), slot, part);
				break;
			}
		tree.chunk(13).upload(block.data(), block.size());

		const OctreeStats before = tree.memory_usage();
		EXPECT_THROW(tree.rebuild(), VulkanError);
		const OctreeStats after = tree.memory_usage();
		EXPECT_EQ(after.bytes, before.bytes);
		EXPECT_EQ(after.chunks, before.chunks);
		for (uint32_t level = 0; level < fmt::kLevelCount; ++level)
			EXPECT_EQ(after.blocks[level], before.blocks[level]) << level;
	}
}
