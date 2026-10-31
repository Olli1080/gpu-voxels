// CPU only: the probabilistic part of the dynamic octree reference (slice S6d-3a). The expected values of the first cases were derived BY HAND
// from the CUDA rules (see the header of octree_dynamic_reference.h), not by running the reference.
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

	std::vector<ref::Node> masked(std::vector<ref::Node> dump)
	{
		for (auto& n : dump)
			// the build gives an occupied leaf the status STATIC only, an insert always UNKNOWN (CUDA setOccupied vs _updateStatus)
			n.status &= uint8_t(~(ref::status::STATIC | ref::status::DYNAMIC | ref::status::OCCUPANCY_MASK));
		return dump;
	}
}

TEST(DynamicOctreeReferenceProb, OccupiedVoxelIntoTheEmptyTree)
{
	Dyn tree(ref::Kind::Probabilistic);
	tree.insert(0, Dyn::Datum::Occupied);
	tree.propagate();
	const auto dump = tree.dump();
	ASSERT_EQ(dump.size(), 1u + 14u * 8u);
	// hand-derived: UNKNOWN 2, PART 8, LAST_LEVEL 16, DYNAMIC 128; the root keeps its own status (no DYNAMIC)
	expect_node(dump, 14, 0, 10, 127);
	for (uint32_t level = 13; level >= 2; --level)
	{
		expect_node(dump, level, 0, 138, 127);
		for (uint64_t s = 1; s < 8; ++s)
			expect_node(dump, level, s, 130, -128);
	}
	expect_node(dump, 1, 0, 154, 127);
	for (uint64_t s = 1; s < 8; ++s)
		expect_node(dump, 1, s, 146, -128);
	expect_node(dump, 0, 0, 130, 127);
	for (uint64_t s = 1; s < 8; ++s)
		expect_node(dump, 0, s, 130, -128);
	EXPECT_FALSE(tree.has_pending_updates());
}

TEST(DynamicOctreeReferenceProb, ExistingLeavesAreUpdatedAndSaturate)
{
	Dyn tree(ref::Kind::Probabilistic);
	tree.insert(0, Dyn::Datum::Occupied);
	tree.propagate();
	tree.insert(0, Dyn::Datum::Occupied);
	tree.propagate();
	expect_node(tree.dump(), 0, 0, 130, 127); // 127 + 127 saturates at MAX
	tree.insert(0, Dyn::Datum::Free);
	tree.propagate();
	auto dump = tree.dump();
	expect_node(dump, 0, 0, 130, 0); // 127 - 127
	expect_node(dump, 1, 0, 154, 0); // the maximum of the leaves follows
	expect_node(dump, 14, 0, 10, 0);
	tree.insert(0, Dyn::Datum::Free);
	tree.insert(0, Dyn::Datum::Free);
	tree.propagate();
	expect_node(tree.dump(), 0, 0, 130, -127); // 0 - 127 - 127 saturates at MIN, not at UNKNOWN (-128)

	// a new free leaf is SET to MIN, a second free update keeps it there
	tree.insert(1, Dyn::Datum::Free);
	tree.propagate();
	expect_node(tree.dump(), 0, 1, 130, -127);
	tree.insert(1, Dyn::Datum::Free);
	tree.propagate();
	expect_node(tree.dump(), 0, 1, 130, -127);
}

TEST(DynamicOctreeReferenceProb, UnknownLeavesCountAsZeroWhenUpdated)
{
	// a built tree: the leaves next to the occupied voxel are UNKNOWN fillers (-128)
	const ref::OctreeReference built(ref::Kind::Probabilistic, { { 0, 0, 0 } });
	Dyn tree(built.dump(), ref::Kind::Probabilistic);
	expect_node(tree.dump(), 0, 1, 66, -128);
	tree.insert(1, Dyn::Datum::Occupied);
	tree.propagate();
	expect_node(tree.dump(), 0, 1, 66, 127); // 0 + 127: UNKNOWN is treated as 0, the status is UNKNOWN, STATIC stays
	tree.insert(2, Dyn::Datum::Free);
	tree.propagate();
	expect_node(tree.dump(), 0, 2, 66, -127); // 0 - 127
}

TEST(DynamicOctreeReferenceProb, InsertBelowAUniformNodeExpandsItsOccupancy)
{
	// a full 2^3 block of occupied voxels: the level 1 node is uniform (occupancy 127, no PART) and keeps its leaf block
	std::vector<ref::Voxel> cube;
	for (uint32_t i = 0; i < 8; ++i)
		cube.push_back({ i & 1, (i >> 1) & 1, (i >> 2) & 1 });
	const ref::OctreeReference built(ref::Kind::Probabilistic, cube);
	Dyn tree(built.dump(), ref::Kind::Probabilistic);
	ASSERT_EQ(node_of(tree.dump(), 1, 0).status & ref::status::PART, 0);
	tree.insert(0, Dyn::Datum::Free);
	tree.propagate();
	const auto dump = tree.dump();
	// the walk ends on the level 1 node without PART: a new leaf block replaces the old one; its invalid leaves copy 127, voxel 0 is SET to -127
	expect_node(dump, 0, 0, 130, -127);
	for (uint64_t s = 1; s < 8; ++s)
		expect_node(dump, 0, s, 130, 127);
	EXPECT_EQ(node_of(dump, 1, 0).occupancy, 127);
	EXPECT_NE(node_of(dump, 1, 0).status & ref::status::PART, 0) << "the leaves differ now";
	EXPECT_EQ(tree.garbage_blocks(0), 1u);
}

TEST(DynamicOctreeReferenceProb, InsertThenPropagateEqualsTheFreshBuildUpToStaticDynamicAndOccupancyBits)
{
	std::mt19937_64 rng(2027);
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
			for (uint32_t i = 0; i < 8; ++i)
				voxels.push_back({ base + (i & 1), base + ((i >> 1) & 1), base + ((i >> 2) & 1) });
		std::vector<uint64_t> ids;
		for (const auto& v : voxels)
			ids.push_back(ref::morton(v.x, v.y, v.z));
		std::sort(ids.begin(), ids.end());
		ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
		std::shuffle(ids.begin(), ids.end(), rng);

		const auto expected = ref::OctreeReference(ref::Kind::Probabilistic, voxels).dump();
		Dyn at_end(ref::Kind::Probabilistic), each_time(ref::Kind::Probabilistic);
		for (uint64_t id : ids)
		{
			at_end.insert(id, Dyn::Datum::Occupied);
			each_time.insert(id, Dyn::Datum::Occupied);
			each_time.propagate();
		}
		at_end.propagate();
		SCOPED_TRACE("round " + std::to_string(round));
		EXPECT_FALSE(at_end.has_pending_updates());
		EXPECT_TRUE(masked(at_end.dump()) == masked(expected)) << "one propagate after all inserts";
		EXPECT_TRUE(masked(each_time.dump()) == masked(expected)) << "propagate after every insert";
		// in the probabilistic rule the status bits do not depend on the moment of the propagate (nothing is copied from the parent)
		EXPECT_TRUE(at_end.dump() == each_time.dump());
	}
}
