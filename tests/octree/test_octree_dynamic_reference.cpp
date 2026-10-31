// CPU only: tests of the dynamic octree reference (slice S6d-1). The expected trees of the first cases were derived BY HAND from the
// CUDA rules (see the header of octree_dynamic_reference.h), not by running the reference; the random cases compare the occupancy
// structure after inserts with the independent recursive build of OctreeReference.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "../reference/octree_dynamic_reference.h"

namespace ref = gvl_reference::octree;
using Dyn = ref::DynamicOctreeReference;

namespace
{
	using NodeKey = std::pair<uint32_t, uint64_t>;

	uint8_t status_of(const std::vector<ref::Node>& dump, uint32_t level, uint64_t id)
	{
		for (const auto& n : dump)
			if (n.level == level && n.id == id)
				return n.status;
		ADD_FAILURE() << "node (" << level << ", " << id << ") is not in the dump";
		return 0;
	}

	bool has_node(const std::vector<ref::Node>& dump, uint32_t level, uint64_t id)
	{
		return std::any_of(dump.begin(), dump.end(), [&](const ref::Node& n) { return n.level == level && n.id == id; });
	}

	std::vector<ref::Node> without_static_dynamic(std::vector<ref::Node> dump)
	{
		for (auto& n : dump)
			n.status &= uint8_t(~(ref::status::STATIC | ref::status::DYNAMIC));
		return dump;
	}
}

TEST(DynamicOctreeReference, OccupiedVoxelIntoTheEmptyTree)
{
	Dyn tree;
	tree.insert(0, Dyn::Datum::Occupied);
	EXPECT_TRUE(tree.has_pending_updates());
	tree.propagate();
	EXPECT_FALSE(tree.has_pending_updates());
	const auto dump = tree.dump();
	EXPECT_EQ(dump.size(), 1u + 14u * 8u) << "the root and one block on each of the levels 13..0";

	// hand-derived: OCCUPIED 4, UNKNOWN 2, PART 8, LAST_LEVEL 16, DYNAMIC 128
	EXPECT_EQ(status_of(dump, 14, 0), 142) << "root: occupied, unknown, PART, DYNAMIC (the inserted leaf is DYNAMIC)";
	for (uint32_t level = 13; level >= 2; --level)
	{
		EXPECT_EQ(status_of(dump, level, 0), 142) << "level " << level;
		for (uint64_t s = 1; s < 8; ++s)
			EXPECT_EQ(status_of(dump, level, s), 2) << "sibling " << s << " of level " << level << " took the status UNKNOWN of the empty tree";
	}
	EXPECT_EQ(status_of(dump, 1, 0), 158) << "142 and LAST_LEVEL";
	for (uint64_t s = 1; s < 8; ++s)
		EXPECT_EQ(status_of(dump, 1, s), 18) << "UNKNOWN | LAST_LEVEL";
	EXPECT_EQ(status_of(dump, 0, 0), 132) << "OCCUPIED | DYNAMIC";
	for (uint64_t s = 1; s < 8; ++s)
		EXPECT_EQ(status_of(dump, 0, s), 2) << "the invalid leaves took UNKNOWN from their parent";
	for (uint32_t level = 0; level < 14; ++level)
		EXPECT_EQ(tree.garbage_blocks(level), 0u);
}

TEST(DynamicOctreeReference, SecondVoxelInTheSameLeafBlockUpdatesTheLeaf)
{
	Dyn tree;
	tree.insert(0, Dyn::Datum::Occupied);
	tree.propagate();
	tree.insert(1, Dyn::Datum::Occupied);
	tree.propagate();
	const auto dump = tree.dump();
	EXPECT_EQ(dump.size(), 1u + 14u * 8u) << "no new blocks";
	EXPECT_EQ(status_of(dump, 0, 1), 4) << "UNKNOWN replaced by OCCUPIED, no DYNAMIC bit: the leaf was not created by an insert";
	EXPECT_EQ(status_of(dump, 0, 0), 132);
	EXPECT_EQ(status_of(dump, 1, 0), 158);
	EXPECT_EQ(status_of(dump, 14, 0), 142);
}

TEST(DynamicOctreeReference, InsertIntoAFullBlockExpandsTheUniformNode)
{
	std::vector<ref::Voxel> cube;
	for (uint32_t z = 0; z < 2; ++z)
		for (uint32_t y = 0; y < 2; ++y)
			for (uint32_t x = 0; x < 2; ++x)
				cube.push_back({ x, y, z });
	const ref::OctreeReference built(ref::Kind::Deterministic, cube);
	const auto before = built.dump();
	ASSERT_EQ(status_of(before, 1, 0), 84) << "OCCUPIED | STATIC | LAST_LEVEL, no PART: a uniform node that keeps its children";

	Dyn tree(before);
	tree.insert(0, Dyn::Datum::Free);
	tree.propagate();
	const auto dump = tree.dump();
	// hand-derived: FREE 1, OCCUPIED 4, PART 8, STATIC 64, DYNAMIC 128
	EXPECT_EQ(status_of(dump, 0, 0), 129) << "FREE | DYNAMIC";
	for (uint64_t s = 1; s < 8; ++s)
		EXPECT_EQ(status_of(dump, 0, s), 68) << "the other seven voxels stay occupied (expanded from the uniform node), STATIC";
	EXPECT_EQ(status_of(dump, 1, 0), 221) << "FREE | OCCUPIED | PART | STATIC | DYNAMIC | LAST_LEVEL";
	for (uint32_t level = 2; level <= 14; ++level)
		EXPECT_EQ(status_of(dump, level, 0), 207) << "level " << level << ": all three occupancy bits, PART, STATIC, DYNAMIC";
	EXPECT_EQ(tree.garbage_blocks(0), 1u) << "the old leaf block lost its pointer";
	EXPECT_EQ(dump.size(), before.size()) << "the old block is replaced by the new one in the listing";
}

TEST(DynamicOctreeReference, InsertingIntoAUniformNodeWithoutChildrenCreatesTheBlocksBelow)
{
	// a uniform OCCUPIED node on level 2 without children: build a tree whose level 2 block is a full 4^3 cube? Its level 1 children
	// are uniform nodes, the level 2 node keeps a child block, so use a hand-made start: the root alone, status OCCUPIED, no PART
	std::vector<ref::Node> start = { ref::Node{ 14, 0, uint8_t(ref::status::OCCUPIED | ref::status::STATIC), 0, false } };
	Dyn tree(start);
	tree.insert(0, Dyn::Datum::Free);
	tree.propagate();
	const auto dump = tree.dump();
	EXPECT_EQ(dump.size(), 1u + 14u * 8u);
	// every sibling on the way down expanded to OCCUPIED | STATIC (68), LAST_LEVEL on level 1 (84), the leaves 68
	for (uint32_t level = 13; level >= 2; --level)
		for (uint64_t s = 1; s < 8; ++s)
			EXPECT_EQ(status_of(dump, level, s), 68) << "level " << level;
	for (uint64_t s = 1; s < 8; ++s)
	{
		EXPECT_EQ(status_of(dump, 1, s), 84);
		EXPECT_EQ(status_of(dump, 0, s), 68);
	}
	EXPECT_EQ(status_of(dump, 0, 0), 129);
	EXPECT_EQ(status_of(dump, 14, 0), 205) << "FREE | OCCUPIED | PART | STATIC | DYNAMIC, no LAST_LEVEL";
}

TEST(DynamicOctreeReference, GarbageIsCountedWhereTheStaleBlocksWere)
{
	// a full 4^3 cube: the level 2 node keeps a block of level 1 nodes, each of which keeps a leaf block (1 + 8 blocks)
	std::vector<ref::Voxel> cube;
	for (uint32_t z = 0; z < 4; ++z)
		for (uint32_t y = 0; y < 4; ++y)
			for (uint32_t x = 0; x < 4; ++x)
				cube.push_back({ x, y, z });
	Dyn tree(ref::OctreeReference(ref::Kind::Deterministic, cube).dump());
	tree.insert(ref::morton(0, 0, 0), Dyn::Datum::Free);
	tree.propagate();
	// the walk stops at the uniform level 2 node (no PART): its whole stale subtree is dropped, the level 1 block and the 8 leaf blocks
	EXPECT_EQ(tree.garbage_blocks(1), 1u);
	EXPECT_EQ(tree.garbage_blocks(0), 8u);
	EXPECT_EQ(tree.garbage_blocks(2), 0u);
}

TEST(DynamicOctreeReference, InsertThenPropagateEqualsTheFreshBuildUpToStaticAndDynamicBits)
{
	std::mt19937_64 rng(2026);
	for (int round = 0; round < 40; ++round)
	{
		const uint32_t extent = round % 3 == 0 ? 6 : (round % 3 == 1 ? 40 : 3000);
		const uint32_t base = round % 2 ? 0 : 5000;
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<ref::Voxel> voxels;
		const size_t count = 1 + rng() % 60;
		for (size_t i = 0; i < count; ++i)
			voxels.push_back({ base + d(rng), base + d(rng), base + d(rng) });
		if (round % 4 == 0)
			for (uint32_t i = 0; i < 8; ++i) // a full 2^3 block (uniform node with children)
				voxels.push_back({ base + (i & 1), base + ((i >> 1) & 1), base + ((i >> 2) & 1) });

		std::vector<uint64_t> ids;
		for (const auto& v : voxels)
			ids.push_back(ref::morton(v.x, v.y, v.z));
		std::sort(ids.begin(), ids.end());
		ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
		std::vector<uint64_t> shuffled = ids;
		std::shuffle(shuffled.begin(), shuffled.end(), rng);

		const auto expected = ref::OctreeReference(ref::Kind::Deterministic, voxels).dump();

		Dyn at_end, each_time;
		for (uint64_t id : shuffled)
		{
			at_end.insert(id, Dyn::Datum::Occupied);
			each_time.insert(id, Dyn::Datum::Occupied);
			each_time.propagate();
		}
		at_end.propagate();
		SCOPED_TRACE("round " + std::to_string(round));
		EXPECT_FALSE(at_end.has_pending_updates());
		EXPECT_TRUE(without_static_dynamic(at_end.dump()) == without_static_dynamic(expected)) << "one propagate after all inserts";
		EXPECT_TRUE(without_static_dynamic(each_time.dump()) == without_static_dynamic(expected)) << "propagate after every insert";
		// NOT equal in general (and not asserted): an expanded node copies STATIC / DYNAMIC from its parent as the parent is at that
		// moment, so these two bits depend on when propagate runs. Occupancy, PART and LAST_LEVEL do not (checked above).
	}
}

TEST(DynamicOctreeReference, FreeAfterOccupiedKeepsTheOtherVoxelsAndTheStructure)
{
	Dyn tree;
	for (uint64_t id : { 0ull, 1ull, 9ull, 100ull })
		tree.insert(id, Dyn::Datum::Occupied);
	tree.propagate();
	tree.insert(1, Dyn::Datum::Free);
	tree.propagate();
	const auto dump = tree.dump();
	EXPECT_EQ(status_of(dump, 0, 1) & 7, 1) << "voxel 1 is FREE now";
	EXPECT_EQ(status_of(dump, 0, 0) & 7, 4);
	EXPECT_EQ(status_of(dump, 0, 9) & 7, 4);
	EXPECT_EQ(status_of(dump, 0, 100) & 7, 4);
	EXPECT_TRUE(has_node(dump, 0, 0) && has_node(dump, 0, 100));
	EXPECT_FALSE(tree.has_pending_updates());
	EXPECT_EQ(status_of(dump, 14, 0) & 7, 7) << "the root sees free, occupied and unknown space";
}
