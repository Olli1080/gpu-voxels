// CPU only: target levels, the hard / soft mode and UPDATE_SUBTREE of the dynamic octree reference (slice S6d-3b). The expected values of the
// first cases were derived BY HAND from the CUDA rules (see the header of octree_dynamic_reference.h), not by running the reference.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "../reference/octree_dynamic_reference.h"

namespace ref = gvl_reference::octree;
using Dyn = ref::DynamicOctreeReference;

namespace
{
	const ref::Node& node_of(const std::vector<ref::Node>& dump, uint32_t level, uint64_t id)
	{
		for (const auto& n : dump)
			if (n.level == level && n.id == id)
				return n;
		static const ref::Node missing{ 99, 0, 0, 0, false };
		ADD_FAILURE() << "node (" << level << ", " << id << ") is not in the dump";
		return missing;
	}

	void expect_node(const std::vector<ref::Node>& dump, uint32_t level, uint64_t id, int status, int occupancy)
	{
		const auto& n = node_of(dump, level, id);
		EXPECT_EQ(int(n.status), status) << "status of (" << level << ", " << id << ")";
		EXPECT_EQ(int(n.occupancy), occupancy) << "occupancy of (" << level << ", " << id << ")";
	}

	/// A probabilistic tree built from one voxel at the origin: leaf (0, 0) occupied, everything else filler.
	Dyn built_prob_tree()
	{
		return Dyn(ref::OctreeReference(ref::Kind::Probabilistic, { { 0, 0, 0 } }).dump(), ref::Kind::Probabilistic);
	}
}

TEST(DynamicOctreeReferenceLevels, FreeNodeOfLevel3IntoTheEmptyTree)
{
	Dyn tree;
	tree.insert(0, Dyn::Datum::Free, 3);
	tree.propagate();
	const auto dump = tree.dump();
	ASSERT_EQ(dump.size(), 1u + 11u * 8u) << "the root and one block on each of the levels 13..3";
	// hand-derived: FREE 1, UNKNOWN 2, PART 8, DYNAMIC 128
	expect_node(dump, 3, 0, 129, 0); // FREE | DYNAMIC, no PART, no children
	EXPECT_FALSE(node_of(dump, 3, 0).has_children);
	for (uint64_t s = 1; s < 8; ++s)
		expect_node(dump, 3, s, 2, 0);
	for (uint32_t level = 4; level <= 14; ++level)
	{
		expect_node(dump, level, 0, 139, 0); // FREE | UNKNOWN | PART | DYNAMIC
		EXPECT_TRUE(node_of(dump, level, 0).has_children);
		if (level < 14)
			for (uint64_t s = 1; s < 8; ++s)
				expect_node(dump, level, s, 2, 0);
	}
	EXPECT_FALSE(tree.has_pending_updates());
}

TEST(DynamicOctreeReferenceLevels, HardInsertOnANodeWithPartCutsTheSubtreeButKeepsItReachable)
{
	// two voxels in one 4^3 box: the level 2 node of the box has PART and a child block
	const ref::OctreeReference built(ref::Kind::Deterministic, { { 0, 0, 0 }, { 3, 3, 3 } });
	Dyn tree(built.dump());
	const auto before = tree.dump();
	ASSERT_NE(node_of(before, 2, 0).status & ref::status::PART, 0);
	tree.insert(0, Dyn::Datum::Free, 2);
	tree.propagate();
	const auto after = tree.dump();
	EXPECT_EQ(node_of(after, 2, 0).status & 15, 1) << "FREE only, PART cleared";
	EXPECT_TRUE(node_of(after, 2, 0).has_children) << "the child block is still reachable (stale)";
	EXPECT_EQ(after.size(), before.size());
	// the stale children keep exactly their old states
	for (const auto& n : before)
		if (n.level < 2)
			EXPECT_TRUE(node_of(after, n.level, n.id) == n) << "stale node (" << n.level << ", " << n.id << ")";
	EXPECT_NE(node_of(after, 14, 0).status & ref::status::PART, 0);
	EXPECT_EQ(node_of(after, 14, 0).status & ref::status::OCCUPIED, 0) << "both voxels lay in the cut box: the root no longer sees them";
	EXPECT_NE(node_of(after, 14, 0).status & ref::status::FREE, 0) << "it sees the free box";
	for (uint32_t level = 0; level <= 14; ++level)
		EXPECT_EQ(tree.garbage_blocks(std::min(level, 13u)), 0u);
	EXPECT_FALSE(tree.has_pending_updates());

	// an insert below the cut node replaces the stale block: now it is garbage
	tree.insert(ref::morton(1, 1, 1), Dyn::Datum::Occupied, 0);
	tree.propagate();
	EXPECT_EQ(tree.garbage_blocks(1), 1u);
	EXPECT_EQ(tree.garbage_blocks(0), 2u) << "the two leaf blocks below the old level 1 nodes";
}

TEST(DynamicOctreeReferenceLevels, ProbabilisticNodeOfLevel2IntoTheEmptyTree)
{
	Dyn tree(ref::Kind::Probabilistic);
	tree.insert(0, Dyn::Datum::Occupied, 2);
	tree.propagate();
	const auto dump = tree.dump();
	ASSERT_EQ(dump.size(), 1u + 12u * 8u) << "the root and one block on each of the levels 13..2";
	// hand-derived: UNKNOWN 2, PART 8, DYNAMIC 128
	expect_node(dump, 2, 0, 130, 127); // set, not added, no PART
	for (uint64_t s = 1; s < 8; ++s)
		expect_node(dump, 2, s, 130, -128);
	expect_node(dump, 3, 0, 138, 127);
	expect_node(dump, 13, 0, 138, 127);
	expect_node(dump, 14, 0, 10, 127);
}

TEST(DynamicOctreeReferenceLevels, ProbabilisticHardInsertOnALevel1NodeCutsItsLeaves)
{
	Dyn tree = built_prob_tree();
	const auto before = tree.dump();
	expect_node(before, 1, 0, 90, 127); // PART | LAST_LEVEL | STATIC | UNKNOWN
	tree.insert(0, Dyn::Datum::Free, 1);
	tree.propagate();
	const auto after = tree.dump();
	expect_node(after, 1, 0, 82, -127); // PART cleared, occupancy SET to -127
	EXPECT_TRUE(node_of(after, 1, 0).has_children);
	expect_node(after, 0, 0, 64, 127); // the stale leaf keeps its state
	expect_node(after, 2, 0, 74, -127); // max(-127, fillers -128), not all equal: PART stays
	expect_node(after, 14, 0, 74, -127);
}

TEST(DynamicOctreeReferenceLevels, ProbabilisticSoftInsertOnALevel1NodeUpdatesItsLeaves)
{
	Dyn tree = built_prob_tree();
	tree.insert(0, Dyn::Datum::Free, 1, Dyn::Mode::Soft);
	tree.propagate();
	const auto after = tree.dump();
	// hand-derived: the node keeps PART and is SET to -127, its leaves are UPDATED with -127: 127 - 127 = 0, -128 - 127 -> -127 (clamp)
	expect_node(after, 0, 0, 64, 0);
	for (uint64_t s = 1; s < 8; ++s)
		expect_node(after, 0, s, 66, -127);
	expect_node(after, 1, 0, 90, 0); // recomputed: the maximum of the leaves, PART again because they differ
	expect_node(after, 2, 0, 74, 0);
	expect_node(after, 14, 0, 74, 0);
	EXPECT_FALSE(tree.has_pending_updates());
}

TEST(DynamicOctreeReferenceLevels, ProbabilisticSoftInsertOnALevel2NodeReachesTheLeavesThroughThePartChildren)
{
	Dyn tree = built_prob_tree();
	tree.insert(0, Dyn::Datum::Free, 2, Dyn::Mode::Soft);
	tree.propagate();
	const auto after = tree.dump();
	// hand-derived: (2,0) is SET to -127 and keeps PART; its children: (1,0) has PART and takes -127 (override), the fillers (1,1..7) are
	// UPDATED: -128 - 127 -> -127; the leaves of (1,0) are UPDATED with its occupancy: 127 - 127 = 0 and -128 - 127 -> -127
	expect_node(after, 0, 0, 64, 0);
	for (uint64_t s = 1; s < 8; ++s)
		expect_node(after, 0, s, 66, -127);
	expect_node(after, 1, 0, 90, 0);
	for (uint64_t s = 1; s < 8; ++s)
		expect_node(after, 1, s, 82, -127); // fillers: UNKNOWN | STATIC | LAST_LEVEL
	expect_node(after, 2, 0, 74, 0);
	expect_node(after, 14, 0, 74, 0);
}

TEST(DynamicOctreeReferenceLevels, TheSubtreeFlagIsPassedDownThroughSeveralLevels)
{
	Dyn tree = built_prob_tree();
	tree.insert(0, Dyn::Datum::Free, 3, Dyn::Mode::Soft);
	tree.propagate();
	const auto after = tree.dump();
	// hand-derived: (3,0) is SET to -127; (2,0) has PART and takes -127, the fillers (2,1..7) are UPDATED (-128 - 127 -> -127); one level
	// down (1,0) takes -127 and the fillers (1,1..7) are updated; the leaves are updated with -127: 127 -> 0 and -128 -> -127; then every
	// node on the path is recomputed from below: 0
	expect_node(after, 0, 0, 64, 0);
	for (uint64_t s = 1; s < 8; ++s)
	{
		expect_node(after, 0, s, 66, -127);
		expect_node(after, 1, s, 82, -127);
		expect_node(after, 2, s, 66, -127);
	}
	expect_node(after, 1, 0, 90, 0);
	expect_node(after, 2, 0, 74, 0);
	expect_node(after, 3, 0, 74, 0);
	expect_node(after, 3, 1, 66, -128); // a sibling of the node on level 3 was not touched
	expect_node(after, 14, 0, 74, 0);
}

TEST(DynamicOctreeReferenceLevels, AFlagLeftBySoftThenHardInsertDoesNotUpdateTheNewSubtree)
{
	Dyn tree = built_prob_tree();
	tree.insert(0, Dyn::Datum::Free, 1, Dyn::Mode::Soft); // sets UPDATE_SUBTREE on (1,0), which has PART
	tree.insert(0, Dyn::Datum::Free, 1, Dyn::Mode::Hard); // cuts the node before the first propagate
	tree.propagate();
	expect_node(tree.dump(), 1, 0, 82, -127);
	tree.insert(1, Dyn::Datum::Occupied, 0); // a new leaf block below the cut node
	tree.propagate();
	const auto after = tree.dump();
	// hand-derived: the new leaves are SET / expanded and NOT updated with the occupancy of the node: voxel 1 is 127, the invalid ones copy -127
	expect_node(after, 0, 1, 130, 127);
	for (uint64_t s = 0; s < 8; ++s)
		if (s != 1)
			expect_node(after, 0, s, 130, -127);
	expect_node(after, 1, 0, 82 + 8, 127); // PART again because the leaves differ
	EXPECT_EQ(tree.garbage_blocks(0), 1u) << "the stale leaf block";
}

TEST(DynamicOctreeReferenceLevels, ASoftInsertOnANodeWithoutPartLeavesNoSubtreeFlag)
{
	Dyn tree = built_prob_tree();
	tree.insert(0, Dyn::Datum::Free, 2, Dyn::Mode::Soft); // (2,0) has PART: not this case; the node (2,1) next to it has none:
	tree.propagate();
	tree.insert(ref::morton(4, 0, 0), Dyn::Datum::Free, 2, Dyn::Mode::Soft);
	tree.insert(ref::morton(4, 0, 0), Dyn::Datum::Occupied, 0); // new blocks below (2,1) before the next propagate
	tree.propagate();
	const auto after = tree.dump();
	// hand-derived: (2,1) was SET to -127 without a subtree flag; the new blocks below it expand -127, voxel 64 is SET to 127 (no update)
	expect_node(after, 0, 64, 130, 127);
	for (uint64_t s = 65; s < 72; ++s)
		expect_node(after, 0, s, 130, -127);
	expect_node(after, 1, 8, 154, 127);
	for (uint64_t s = 9; s < 16; ++s)
		expect_node(after, 1, s, 146, -127);
	EXPECT_EQ(node_of(after, 2, 1).occupancy, 127);
	EXPECT_NE(node_of(after, 2, 1).status & ref::status::PART, 0);
}

TEST(DynamicOctreeReferenceLevels, TheDeterministicSoftModeCutsLikeTheHardMode)
{
	const ref::OctreeReference built(ref::Kind::Deterministic, { { 0, 0, 0 }, { 3, 3, 3 } });
	Dyn hard(built.dump()), soft(built.dump());
	hard.insert(0, Dyn::Datum::Free, 2, Dyn::Mode::Hard);
	soft.insert(0, Dyn::Datum::Free, 2, Dyn::Mode::Soft);
	hard.propagate();
	soft.propagate();
	EXPECT_TRUE(hard.dump() == soft.dump());
}

TEST(DynamicOctreeReferenceLevels, RandomInsertsKeepTheProbabilisticInvariants)
{
	std::mt19937_64 rng(2028);
	for (int round = 0; round < 30; ++round)
	{
		std::uniform_int_distribution<uint32_t> d(0, 15);
		std::vector<ref::Voxel> start;
		for (int i = 0; i < 40; ++i)
			start.push_back({ d(rng), d(rng), d(rng) });
		Dyn tree(ref::OctreeReference(ref::Kind::Probabilistic, start).dump(), ref::Kind::Probabilistic);
		for (int step = 0; step < 6; ++step)
		{
			const uint32_t level = uint32_t(rng() % 5);
			const uint64_t id = ref::morton(d(rng), d(rng), d(rng));
			tree.insert(id, rng() % 2 ? Dyn::Datum::Occupied : Dyn::Datum::Free, level, rng() % 2 ? Dyn::Mode::Hard : Dyn::Mode::Soft);
			if (rng() % 2)
				tree.propagate();
		}
		tree.propagate();
		EXPECT_FALSE(tree.has_pending_updates());
		// every node reachable through PART nodes from the root: its occupancy is the maximum of its children and it is not uniform (stale
		// subtrees below a cut node are not recomputed, a state CUDA leaves behind as well, they are not checked)
		const auto dump = tree.dump();
		std::vector<std::pair<uint32_t, uint64_t>> stack = { { 14u, 0u } };
		while (!stack.empty())
		{
			const auto [level, id] = stack.back();
			stack.pop_back();
			const auto& n = node_of(dump, level, id);
			if (level == 0 || !(n.status & ref::status::PART))
				continue;
			int highest = -128;
			bool same = true;
			std::vector<ref::Node> kids;
			for (const auto& k : dump)
				if (k.level == level - 1 && k.id / 8 == id)
					kids.push_back(k);
			ASSERT_EQ(kids.size(), 8u) << "node (" << level << ", " << id << ")";
			for (const auto& k : kids)
				highest = std::max(highest, int(k.occupancy));
			for (const auto& k : kids)
				same = same && k.occupancy == highest && !(k.status & ref::status::PART);
			std::string detail;
			for (const auto& k : kids)
				detail += " [" + std::to_string(int(k.occupancy)) + "/" + std::to_string(int(k.status)) + "]";
			EXPECT_EQ(int(n.occupancy), highest) << "round " << round << " node (" << level << ", " << id << ") status " << int(n.status) << " kids" << detail;
			EXPECT_FALSE(same) << "a node with PART whose children are uniform, round " << round;
			for (const auto& k : kids)
				stack.push_back({ k.level, k.id });
		}
	}
}
