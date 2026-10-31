// Slice S6d-2: insert (hard set at level 0) and propagate of the deterministic octree on the GPU against the independent dynamic reference
// (tests/reference/octree_dynamic_reference.h): after every step the whole node list (status incl. STATIC / DYNAMIC, child blocks) must be
// equal, check_tree must pass, no node may keep NEEDS_UPDATE after propagate and the garbage blocks per level must match.
#include "vulkan_fixture.h"

#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>

#include "../reference/octree_dynamic_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference::octree;
using Dyn = ref::DynamicOctreeReference;

namespace
{
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

	std::vector<ref::Voxel> to_ref(const std::vector<OctreeVoxel>& v)
	{
		std::vector<ref::Voxel> out;
		for (const auto& x : v)
			out.push_back({ x.x, x.y, x.z });
		return out;
	}

	/// Sorted distinct ids of the valid voxels (what the insert sees).
	std::vector<uint64_t> ids_of(const std::vector<OctreeVoxel>& voxels)
	{
		std::set<uint64_t> ids;
		for (const auto& v : voxels)
			if (v.x < 16384 && v.y < 16384 && v.z < 16384)
				ids.insert(ref::morton(v.x, v.y, v.z));
		return { ids.begin(), ids.end() };
	}

	/// Empty string if the dumps are equal, else the first difference.
	std::string difference(const std::vector<OctreeNode>& gpu, const std::vector<ref::Node>& expected)
	{
		if (gpu.size() != expected.size())
			return "node counts differ: GPU " + std::to_string(gpu.size()) + ", reference " + std::to_string(expected.size());
		for (size_t i = 0; i < gpu.size(); ++i)
		{
			const auto& a = gpu[i];
			const auto& b = expected[i];
			if (a.level != b.level || a.id != b.id || a.status != b.status || a.has_children != b.has_children)
				return "node " + std::to_string(i) + ": GPU (level " + std::to_string(a.level) + ", id " + std::to_string(a.id) + ", status " + std::to_string(a.status) +
					(a.has_children ? ", children" : "") + ") reference (level " + std::to_string(b.level) + ", id " + std::to_string(b.id) + ", status " +
					std::to_string(b.status) + (b.has_children ? ", children" : "") + ")";
		}
		return {};
	}

	void expect_same(const Octree& tree, const Dyn& model, const std::string& what)
	{
		EXPECT_EQ(difference(tree.dump(), model.dump()), "") << what;
		EXPECT_NO_THROW(tree.check_tree()) << what;
		EXPECT_EQ(tree.pending_update_count(), 0u) << what << ": NEEDS_UPDATE left behind";
		const auto stats = tree.memory_usage();
		for (uint32_t level = 0; level < 14; ++level)
			EXPECT_EQ(stats.garbage_blocks[level], model.garbage_blocks(level)) << what << ", garbage blocks of level " << level;
	}

	void insert_both(Octree& tree, Dyn& model, const std::vector<OctreeVoxel>& voxels, OctreeDatum datum, bool propagate_each)
	{
		tree.insert(voxels, datum, propagate_each);
		for (uint64_t id : ids_of(voxels))
			model.insert(id, datum == OctreeDatum::Occupied ? Dyn::Datum::Occupied : Dyn::Datum::Free);
		if (propagate_each)
			model.propagate();
	}
}

TEST_F(VulkanTest, OctreeInsertIntoTheEmptyTreeMatchesTheReference)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	Dyn model;
	insert_both(tree, model, { { 0, 0, 0 } }, OctreeDatum::Occupied, true);
	expect_same(tree, model, "one voxel");
	EXPECT_EQ(tree.dump().size(), 1u + 14u * 8u);
	const auto dump = tree.dump();
	EXPECT_EQ(dump.front().level, 0u);
	EXPECT_EQ(dump.front().status, 132) << "OCCUPIED | DYNAMIC, the value derived by hand in the reference tests";
	EXPECT_EQ(dump.back().status, 142);

	insert_both(tree, model, { { 1, 0, 0 } }, OctreeDatum::Occupied, true);
	expect_same(tree, model, "second voxel in the same leaf block");
	EXPECT_EQ(tree.memory_usage().chunks, 15u) << "the second insert needed no new block";

	insert_both(tree, model, { { 16383, 16383, 16383 }, { 8000, 3, 9000 } }, OctreeDatum::Occupied, true);
	expect_same(tree, model, "far voxels");
	EXPECT_GT(tree.memory_usage().chunks, 15u);
}

TEST_F(VulkanTest, OctreeInsertSequencesMatchTheReference)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(2031);
	for (int round = 0; round < 12; ++round)
	{
		SCOPED_TRACE("round " + std::to_string(round));
		const uint32_t extent = round % 3 == 0 ? 12 : (round % 3 == 1 ? 40 : 600);
		const uint32_t base = round % 2 ? 100 : 5000;
		const bool start_from_build = round % 4 != 0;
		Octree tree(ctx(), OctreeKind::Deterministic);
		Dyn model;
		if (start_from_build)
		{
			auto start = random_voxels(rng, 150, extent, base, base, base);
			if (round % 2 == 0)
			{
				const auto block = cube(base, base, base, 2);
				start.insert(start.end(), block.begin(), block.end());
			}
			tree.build(start);
			model = Dyn(ref::OctreeReference(ref::Kind::Deterministic, to_ref(start)).dump());
			expect_same(tree, model, "after the build");
		}
		const bool propagate_each = round % 5 != 0;
		for (int step = 0; step < 5; ++step)
		{
			SCOPED_TRACE("step " + std::to_string(step));
			auto batch = random_voxels(rng, 40 + rng() % 120, extent, base, base, base);
			if (step % 2 == 0) // voxels around the existing ones: updates of existing leaves and new blocks next to old ones
			{
				auto more = random_voxels(rng, 30, extent + 8, base, base, base);
				batch.insert(batch.end(), more.begin(), more.end());
			}
			batch.push_back(batch.front()); // a duplicate inside the batch
			const OctreeDatum datum = step % 3 == 2 ? OctreeDatum::Free : OctreeDatum::Occupied;
			insert_both(tree, model, batch, datum, propagate_each);
			if (propagate_each)
				expect_same(tree, model, "after the insert");
			else if (step % 2 == 1)
			{
				tree.propagate();
				model.propagate();
				expect_same(tree, model, "after the explicit propagate");
			}
		}
		tree.propagate();
		model.propagate();
		expect_same(tree, model, "at the end");
	}
}

TEST_F(VulkanTest, OctreeInsertIntoUniformNodesCountsGarbage)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	const auto start = cube(0, 0, 0, 4);
	tree.build(start);
	Dyn model(ref::OctreeReference(ref::Kind::Deterministic, to_ref(start)).dump());
	expect_same(tree, model, "the 4^3 cube after the build");
	EXPECT_EQ(tree.memory_usage().garbage_blocks[0], 0u);

	insert_both(tree, model, { { 0, 0, 0 } }, OctreeDatum::Free, true);
	expect_same(tree, model, "a free voxel inside the uniform level 2 node");
	const auto stats = tree.memory_usage();
	EXPECT_EQ(stats.garbage_blocks[1], 1u) << "the stale level 1 block";
	EXPECT_EQ(stats.garbage_blocks[0], 8u) << "and its 8 leaf blocks";

	insert_both(tree, model, { { 2, 1, 3 }, { 3, 3, 3 } }, OctreeDatum::Free, true);
	expect_same(tree, model, "more voxels inside the expanded node (updates of existing leaves)");
	// the other voxels of the cube are still occupied: they were expanded from the uniform node
	const auto found = tree.find(std::vector<uint64_t>{ ref::morton(1, 0, 0), ref::morton(0, 0, 0), ref::morton(3, 2, 1) });
	EXPECT_NE(found[0].status & 4, 0);
	EXPECT_EQ(found[1].status & 7, 1);
	EXPECT_NE(found[2].status & 4, 0);
}

TEST_F(VulkanTest, OctreeInsertDropsInvalidVoxelsAndHandlesEmptyInput)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	Dyn model;
	tree.insert(std::vector<OctreeVoxel>{});
	expect_same(tree, model, "an empty insert");
	const auto before = tree.dropped_count();
	insert_both(tree, model, { { 16384, 0, 0 }, { 5, 5, 5 }, { 0, 99999, 0 } }, OctreeDatum::Occupied, true);
	EXPECT_EQ(tree.dropped_count(), before + 2);
	expect_same(tree, model, "invalid voxels dropped, one valid voxel inserted");
	std::vector<uint64_t> bad_ids = { 1ull << 42, (1ull << 42) + 7 };
	tree.insert_ids(bad_ids);
	expect_same(tree, model, "only invalid ids");
	EXPECT_EQ(tree.dropped_count(), before + 4);

	Octree prob(ctx(), OctreeKind::Probabilistic);
	std::mt19937_64 prob_rng(5);
	prob.build(random_voxels(prob_rng, 50, 10));
	const auto prob_before = prob.dump();
	const auto prob_chunks = prob.memory_usage().chunks;
	EXPECT_THROW(prob.insert(std::vector<OctreeVoxel>{ { 1, 1, 1 } }), VulkanError);
	EXPECT_THROW(prob.insert(std::vector<OctreeVoxel>{ { 1, 1, 1 } }, OctreeDatum::Occupied, false), VulkanError);
	EXPECT_TRUE(prob.dump() == prob_before) << "a rejected insert must not touch the probabilistic tree (the deterministic kernels would corrupt it)";
	EXPECT_EQ(prob.memory_usage().chunks, prob_chunks);
	EXPECT_NO_THROW(prob.check_tree());
}

TEST_F(VulkanTest, OctreeAfterInsertsCollidesLikeTheUnionTree)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(2032);
	const auto first = random_voxels(rng, 800, 30, 100, 100, 100);
	const auto second = random_voxels(rng, 800, 30, 112, 112, 112);
	const auto other = random_voxels(rng, 1500, 40, 105, 105, 105);

	Octree grown(ctx(), OctreeKind::Deterministic), fresh(ctx(), OctreeKind::Deterministic), partner(ctx(), OctreeKind::Deterministic);
	grown.build(first);
	grown.insert(second, OctreeDatum::Occupied, true);
	std::vector<OctreeVoxel> all = first;
	all.insert(all.end(), second.begin(), second.end());
	fresh.build(all);
	partner.build(other);

	for (const bool balancer : { false, true })
		for (const uint32_t min_level : { 0u, 1u, 3u })
		{
			OctreeTreeCollisionOptions options;
			options.min_level = min_level;
			options.use_load_balancer = balancer;
			options.mark_collisions = false;
			const uint64_t expected = fresh.collide_with(partner, options);
			EXPECT_GT(expected, 0u);
			EXPECT_EQ(grown.collide_with(partner, options), expected) << "balancer " << balancer << " min_level " << min_level;
		}
	// the occupancy structure equals the fresh build; STATIC / DYNAMIC bits differ and are not part of the comparison
	const auto a = grown.dump(), b = fresh.dump();
	ASSERT_EQ(a.size(), b.size());
	for (size_t i = 0; i < a.size(); ++i)
		ASSERT_TRUE(a[i].level == b[i].level && a[i].id == b[i].id && (a[i].status & 63) == (b[i].status & 63) && a[i].has_children == b[i].has_children) << "node " << i;
}
