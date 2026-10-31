// Slice S6d-3b: inserts on nodes above level 0 (hard and soft mode, UPDATE_SUBTREE) of the deterministic and the probabilistic octree on the GPU
// against the independent dynamic reference (tests/reference/octree_dynamic_reference.h): after every step the whole node list (status,
// occupancy, child blocks, also stale blocks below cut nodes) must be equal, check_tree must pass, no node reachable through PART nodes may keep
// NEEDS_UPDATE after propagate and the garbage blocks per level must match.
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
	constexpr OctreeKind kKinds[] = { OctreeKind::Deterministic, OctreeKind::Probabilistic };
	const char* name(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? "probabilistic" : "deterministic"; }
	ref::Kind to_ref_kind(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? ref::Kind::Probabilistic : ref::Kind::Deterministic; }

	std::vector<OctreeVoxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent, uint32_t base)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<OctreeVoxel> v;
		for (size_t i = 0; i < count; ++i)
			v.push_back({ base + d(rng), base + d(rng), base + d(rng) });
		return v;
	}

	std::vector<ref::Voxel> to_ref(const std::vector<OctreeVoxel>& v)
	{
		std::vector<ref::Voxel> out;
		for (const auto& x : v)
			out.push_back({ x.x, x.y, x.z });
		return out;
	}

	/// Sorted distinct node ids of `level` of the voxels.
	std::vector<uint64_t> node_ids_of(const std::vector<OctreeVoxel>& voxels, uint32_t level)
	{
		std::set<uint64_t> ids;
		for (const auto& v : voxels)
			ids.insert(ref::morton(v.x, v.y, v.z) >> (3 * level));
		return { ids.begin(), ids.end() };
	}

	std::string difference(const std::vector<OctreeNode>& gpu, const std::vector<ref::Node>& expected)
	{
		if (gpu.size() != expected.size())
			return "node counts differ: GPU " + std::to_string(gpu.size()) + ", reference " + std::to_string(expected.size());
		for (size_t i = 0; i < gpu.size(); ++i)
		{
			const auto& a = gpu[i];
			const auto& b = expected[i];
			if (a.level != b.level || a.id != b.id || a.status != b.status || a.occupancy != b.occupancy || a.has_children != b.has_children)
				return "node " + std::to_string(i) + ": GPU (level " + std::to_string(a.level) + ", id " + std::to_string(a.id) + ", status " + std::to_string(a.status) +
					", occupancy " + std::to_string(a.occupancy) + (a.has_children ? ", children" : "") + ") reference (level " + std::to_string(b.level) + ", id " +
					std::to_string(b.id) + ", status " + std::to_string(b.status) + ", occupancy " + std::to_string(b.occupancy) + (b.has_children ? ", children" : "") + ")";
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

	void insert_both(Octree& tree, Dyn& model, uint32_t level, const std::vector<uint64_t>& ids, OctreeDatum datum, OctreeInsertMode mode, bool propagate_each)
	{
		tree.insert_nodes(level, ids, datum, mode, propagate_each);
		for (uint64_t id : ids)
			model.insert(id << (3 * level), datum == OctreeDatum::Occupied ? Dyn::Datum::Occupied : Dyn::Datum::Free, level,
				mode == OctreeInsertMode::Soft ? Dyn::Mode::Soft : Dyn::Mode::Hard);
		if (propagate_each)
			model.propagate();
	}
}

TEST_F(VulkanTest, OctreeInsertOfANodeOfLevel3IntoTheEmptyTreeMatchesTheHandDerivedTree)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	Dyn model(ref::Kind::Deterministic);
	insert_both(tree, model, 3, { 0 }, OctreeDatum::Free, OctreeInsertMode::Hard, true);
	expect_same(tree, model, "a free node of level 3");
	const auto dump = tree.dump();
	ASSERT_EQ(dump.size(), 1u + 11u * 8u);
	// the values derived by hand in the reference tests: FREE | DYNAMIC for the node, FREE | UNKNOWN | PART | DYNAMIC above
	auto status_of = [&](uint32_t level, uint64_t id) { for (const auto& n : dump) if (n.level == level && n.id == id) return int(n.status); return -1; };
	EXPECT_EQ(status_of(3, 0), 129);
	EXPECT_EQ(status_of(3, 1), 2);
	EXPECT_EQ(status_of(4, 0), 139);
	EXPECT_EQ(status_of(14, 0), 139);

	Octree prob(ctx(), OctreeKind::Probabilistic);
	Dyn prob_model(ref::Kind::Probabilistic);
	insert_both(prob, prob_model, 2, { 0 }, OctreeDatum::Occupied, OctreeInsertMode::Hard, true);
	expect_same(prob, prob_model, "an occupied node of level 2");
	const auto pdump = prob.dump();
	ASSERT_EQ(pdump.size(), 1u + 12u * 8u);
	for (const auto& n : pdump)
	{
		if (n.level == 2 && n.id == 0)
		{
			EXPECT_EQ(n.status, 130);
			EXPECT_EQ(n.occupancy, 127);
		}
		if (n.level == 14)
		{
			EXPECT_EQ(n.status, 10);
			EXPECT_EQ(n.occupancy, 127);
		}
	}
}

TEST_F(VulkanTest, OctreeSoftInsertUpdatesTheSubtreeAndHardInsertCutsIt)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const std::vector<OctreeVoxel> one = { { 0, 0, 0 } };
	auto occupancy_of = [](const Octree& tree, uint32_t level, uint64_t id) { for (const auto& n : tree.dump()) if (n.level == level && n.id == id) return int(n.occupancy); return 999; };
	auto status_of = [](const Octree& tree, uint32_t level, uint64_t id) { for (const auto& n : tree.dump()) if (n.level == level && n.id == id) return int(n.status); return -1; };

	// soft on level 1: the leaves are updated with -127 (127 -> 0, -128 -> -127), the node is recomputed (the hand-derived values of the reference tests)
	Octree soft1(ctx(), OctreeKind::Probabilistic);
	soft1.build(one);
	soft1.insert_nodes(1, std::vector<uint64_t>{ 0 }, OctreeDatum::Free, OctreeInsertMode::Soft);
	EXPECT_EQ(occupancy_of(soft1, 0, 0), 0);
	for (uint64_t s = 1; s < 8; ++s)
		EXPECT_EQ(occupancy_of(soft1, 0, s), -127);
	EXPECT_EQ(occupancy_of(soft1, 1, 0), 0);
	EXPECT_EQ(status_of(soft1, 1, 0), 90);
	EXPECT_EQ(occupancy_of(soft1, 14, 0), 0);
	EXPECT_EQ(soft1.pending_update_count(), 0u);
	EXPECT_NO_THROW(soft1.check_tree());

	// soft on level 2 reaches the leaves through the children with PART
	Octree soft2(ctx(), OctreeKind::Probabilistic);
	soft2.build(one);
	soft2.insert_nodes(2, std::vector<uint64_t>{ 0 }, OctreeDatum::Free, OctreeInsertMode::Soft);
	EXPECT_EQ(occupancy_of(soft2, 0, 0), 0);
	EXPECT_EQ(occupancy_of(soft2, 0, 1), -127);
	for (uint64_t s = 1; s < 8; ++s)
		EXPECT_EQ(occupancy_of(soft2, 1, s), -127);
	EXPECT_EQ(occupancy_of(soft2, 2, 0), 0);

	// hard on level 1: PART cleared, occupancy SET to -127, the stale leaves keep their state
	Octree hard1(ctx(), OctreeKind::Probabilistic);
	hard1.build(one);
	hard1.insert_nodes(1, std::vector<uint64_t>{ 0 }, OctreeDatum::Free, OctreeInsertMode::Hard);
	EXPECT_EQ(status_of(hard1, 1, 0), 82);
	EXPECT_EQ(occupancy_of(hard1, 1, 0), -127);
	EXPECT_EQ(occupancy_of(hard1, 0, 0), 127);
	EXPECT_EQ(occupancy_of(hard1, 2, 0), -127);
	EXPECT_NO_THROW(hard1.check_tree());
	for (const auto& n : hard1.dump())
		if (n.level == 1 && n.id == 0)
			EXPECT_TRUE(n.has_children) << "the cut block stays reachable";

	// the deterministic soft mode cuts like the hard mode
	Octree det_soft(ctx(), OctreeKind::Deterministic), det_hard(ctx(), OctreeKind::Deterministic);
	const std::vector<OctreeVoxel> two = { { 0, 0, 0 }, { 3, 3, 3 } };
	det_soft.build(two);
	det_hard.build(two);
	det_soft.insert_nodes(2, std::vector<uint64_t>{ 0 }, OctreeDatum::Free, OctreeInsertMode::Soft);
	det_hard.insert_nodes(2, std::vector<uint64_t>{ 0 }, OctreeDatum::Free, OctreeInsertMode::Hard);
	EXPECT_TRUE(det_soft.dump() == det_hard.dump());
}

TEST_F(VulkanTest, OctreeInsertsOnNodesMatchTheReference)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(2041);
	for (const auto kind : kKinds)
		for (int round = 0; round < 14; ++round)
		{
			SCOPED_TRACE(std::string(name(kind)) + ", round " + std::to_string(round));
			const uint32_t extent = round % 3 == 0 ? 10 : (round % 3 == 1 ? 40 : 300);
			const uint32_t base = round % 2 ? 100 : 5000;
			Octree tree(ctx(), kind);
			Dyn model(to_ref_kind(kind));
			if (round % 4 != 0)
			{
				const auto start = random_voxels(rng, 120, extent, base);
				tree.build(start);
				model = Dyn(ref::OctreeReference(to_ref_kind(kind), to_ref(start)).dump(), to_ref_kind(kind));
				expect_same(tree, model, "after the build");
			}
			const bool propagate_each = round % 5 != 0;
			for (int step = 0; step < 6; ++step)
			{
				SCOPED_TRACE("step " + std::to_string(step));
				const uint32_t level = uint32_t(rng() % 6);
				const auto voxels = random_voxels(rng, 5 + rng() % 40, extent, base);
				const auto ids = node_ids_of(voxels, level);
				const OctreeDatum datum = rng() % 3 ? OctreeDatum::Occupied : OctreeDatum::Free;
				const OctreeInsertMode mode = rng() % 2 ? OctreeInsertMode::Soft : OctreeInsertMode::Hard;
				insert_both(tree, model, level, ids, datum, mode, propagate_each);
				if (propagate_each)
					expect_same(tree, model, "after the insert (level " + std::to_string(level) + ")");
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

TEST_F(VulkanTest, OctreeInsertNodesValidatesTheLevelAndDropsIdsOutsideIt)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		const std::vector<uint64_t> ids = { 1 };
		EXPECT_THROW(tree.insert_nodes(14, ids), VulkanError);
		EXPECT_THROW(tree.insert_nodes(99, ids), VulkanError);
		const auto before = tree.dropped_count();
		// level 12 holds 8^2 = 64 nodes: ids 64.. are outside
		tree.insert_nodes(12, std::vector<uint64_t>{ 5, 64, 1ull << 40 });
		EXPECT_EQ(tree.dropped_count(), before + 2);
		Dyn model(to_ref_kind(kind));
		model.insert(uint64_t(5) << 36, Dyn::Datum::Occupied, 12);
		model.propagate();
		expect_same(tree, model, "only the valid node of level 12 was inserted");
		tree.insert_nodes(3, std::vector<uint64_t>{});
		expect_same(tree, model, "an empty insert changes nothing");
	}
}

TEST_F(VulkanTest, OctreeSubtreeFlagScenariosMatchTheHandDerivedValues)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const std::vector<OctreeVoxel> one = { { 0, 0, 0 } };
	auto make = [&](Octree& tree, Dyn& model)
	{
		tree.build(one);
		model = Dyn(ref::OctreeReference(ref::Kind::Probabilistic, to_ref(one)).dump(), ref::Kind::Probabilistic);
	};
	auto occupancy_of = [](const Octree& tree, uint32_t level, uint64_t id) { for (const auto& n : tree.dump()) if (n.level == level && n.id == id) return int(n.occupancy); return 999; };
	const auto soft = OctreeInsertMode::Soft;
	const auto hard = OctreeInsertMode::Hard;
	const auto free_datum = OctreeDatum::Free;

	{ // the flag of a node on level 3 is passed down through the levels 2 and 1 to the leaves
		Octree tree(ctx(), OctreeKind::Probabilistic);
		Dyn model(ref::Kind::Probabilistic);
		make(tree, model);
		insert_both(tree, model, 3, { 0 }, free_datum, soft, true);
		expect_same(tree, model, "soft insert on level 3");
		EXPECT_EQ(occupancy_of(tree, 0, 0), 0);
		for (uint64_t s = 1; s < 8; ++s)
		{
			EXPECT_EQ(occupancy_of(tree, 0, s), -127);
			EXPECT_EQ(occupancy_of(tree, 1, s), -127);
			EXPECT_EQ(occupancy_of(tree, 2, s), -127);
		}
		EXPECT_EQ(occupancy_of(tree, 3, 0), 0);
		EXPECT_EQ(occupancy_of(tree, 3, 1), -128) << "a sibling of the node was not touched";
	}
	{ // soft then hard on the same node before a propagate: the flag must not survive to update a block created later
		Octree tree(ctx(), OctreeKind::Probabilistic);
		Dyn model(ref::Kind::Probabilistic);
		make(tree, model);
		insert_both(tree, model, 1, { 0 }, free_datum, soft, false);
		insert_both(tree, model, 1, { 0 }, free_datum, hard, false);
		tree.propagate();
		model.propagate();
		expect_same(tree, model, "after soft and hard");
		tree.insert_ids(std::vector<uint64_t>{ 1 });
		model.insert(1, Dyn::Datum::Occupied);
		model.propagate();
		expect_same(tree, model, "a voxel below the cut node");
		EXPECT_EQ(occupancy_of(tree, 0, 1), 127);
		for (uint64_t s = 0; s < 8; ++s)
			if (s != 1)
				EXPECT_EQ(occupancy_of(tree, 0, s), -127);
	}
	{ // a soft insert on a node without PART sets no flag: the block created below it before the propagate is not updated
		Octree tree(ctx(), OctreeKind::Probabilistic);
		Dyn model(ref::Kind::Probabilistic);
		make(tree, model);
		insert_both(tree, model, 2, { 1 }, free_datum, soft, false);
		tree.insert_ids(std::vector<uint64_t>{ 64 }, OctreeDatum::Occupied, false);
		model.insert(64, Dyn::Datum::Occupied);
		tree.propagate();
		model.propagate();
		expect_same(tree, model, "soft insert on a node without PART, then a voxel below it");
		EXPECT_EQ(occupancy_of(tree, 0, 64), 127);
		for (uint64_t s = 65; s < 72; ++s)
			EXPECT_EQ(occupancy_of(tree, 0, s), -127);
		EXPECT_EQ(occupancy_of(tree, 2, 1), 127);
	}
}

