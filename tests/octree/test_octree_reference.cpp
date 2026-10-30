// CPU only (no Vulkan device needed): tests of the host Morton code and of the octree reference itself (slice S6a).
// The reference is the oracle of all later octree tests, so it is checked against hand-derived trees and against a
// brute-force definition (voxel counts inside node boxes) that does not use the reference's own recursion.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include <gpu_voxels/vulkan/Morton.h>
#include <gpu_voxels/vulkan/OctreeFormat.h>

#include "../reference/octree_reference.h"

namespace ref = gvl_reference::octree;
namespace mo = gpu_voxels::vulkan::morton;
namespace fmt = gpu_voxels::vulkan::octree_format;
using gpu_voxels::vulkan::OctreeKind;

namespace
{
	uint64_t pow8(uint32_t level)
	{
		uint64_t r = 1;
		for (uint32_t i = 0; i < level; ++i)
			r *= 8;
		return r;
	}

	std::vector<ref::Voxel> cube(uint32_t x0, uint32_t y0, uint32_t z0, uint32_t side)
	{
		std::vector<ref::Voxel> v;
		for (uint32_t z = 0; z < side; ++z)
			for (uint32_t y = 0; y < side; ++y)
				for (uint32_t x = 0; x < side; ++x)
					v.push_back({ x0 + x, y0 + y, z0 + z });
		return v;
	}

	const ref::Node* find_node(const std::vector<ref::Node>& dump, uint32_t level, uint64_t id)
	{
		for (const auto& n : dump)
			if (n.level == level && n.id == id)
				return &n;
		return nullptr;
	}

	// Brute force oracle of the status rules: number of occupied voxels inside the box of a node.
	size_t voxels_in_box(const ref::OctreeReference& tree, uint32_t level, uint64_t id)
	{
		const auto& ids = tree.voxel_ids();
		const uint64_t first = id * pow8(level);
		const uint64_t last = (id + 1) * pow8(level);
		return size_t(std::lower_bound(ids.begin(), ids.end(), last) - std::lower_bound(ids.begin(), ids.end(), first));
	}

	void check_against_boxes(const ref::OctreeReference& tree)
	{
		const auto dump = tree.dump();
		std::set<std::pair<uint32_t, uint64_t>> present;
		for (const auto& n : dump)
			present.insert({ n.level, n.id });
		size_t inner = 0;
		for (const auto& n : dump)
		{
			const size_t count = voxels_in_box(tree, n.level, n.id);
			const bool full = count == pow8(n.level);
			if (!n.has_children)
			{
				if (count == 0) // filler
				{
					const uint8_t expected = ref::status::UNKNOWN | ref::status::STATIC | (n.level == 1 ? ref::status::LAST_LEVEL : 0);
					EXPECT_EQ(n.status, expected) << n;
					EXPECT_EQ(n.occupancy, tree.kind() == ref::Kind::Probabilistic ? ref::kUnknownOccupancy : 0) << n;
				}
				else
				{
					ASSERT_EQ(n.level, 0u) << n; // only voxels are occupied leaves
					EXPECT_EQ(n.status, tree.kind() == ref::Kind::Deterministic ? (ref::status::OCCUPIED | ref::status::STATIC) : ref::status::STATIC) << n;
					EXPECT_EQ(n.occupancy, tree.kind() == ref::Kind::Probabilistic ? ref::kOccupied : 0) << n;
				}
				continue;
			}
			++inner;
			ASSERT_GT(count, 0u) << n;
			uint8_t expected = ref::status::STATIC | (n.level == 1 ? ref::status::LAST_LEVEL : 0);
			if (tree.kind() == ref::Kind::Deterministic)
				expected |= full ? ref::status::OCCUPIED : (ref::status::OCCUPIED | ref::status::UNKNOWN | ref::status::PART);
			else
				expected |= ref::status::UNKNOWN | (full ? 0 : ref::status::PART);
			EXPECT_EQ(n.status, expected) << n << " voxels " << count << " of " << pow8(n.level);
			EXPECT_EQ(n.occupancy, tree.kind() == ref::Kind::Probabilistic ? ref::kOccupied : 0) << n;
			// the block of 8 children exists
			for (uint32_t c = 0; c < 8; ++c)
				EXPECT_EQ(present.count({ n.level - 1, n.id * 8 + c }), 1u) << n << " child " << c;
		}
		EXPECT_EQ(dump.size(), 1 + 8 * inner);
	}

	std::vector<ref::Voxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent, uint32_t ox = 0, uint32_t oy = 0, uint32_t oz = 0)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<ref::Voxel> v;
		for (size_t i = 0; i < count; ++i)
			v.push_back({ ox + d(rng), oy + d(rng), oz + d(rng) });
		return v;
	}
}

// ---------------------------------------------------------------------------------------------------------------- Morton

TEST(Morton, HostMatchesBitLoopReference)
{
	const uint32_t edge[] = { 0, 1, 2, 3, 1022, 1023, 1024, 1025, 16383, 16384, (1u << 20) - 1 };
	for (uint32_t x : edge)
		for (uint32_t y : edge)
			for (uint32_t z : edge)
			{
				ASSERT_EQ(mo::code60(x, y, z), ref::morton(x, y, z)) << x << " " << y << " " << z;
				const auto c = mo::decode60(ref::morton(x, y, z));
				ASSERT_EQ(c.x, x);
				ASSERT_EQ(c.y, y);
				ASSERT_EQ(c.z, z);
			}
	std::mt19937_64 rng(7);
	for (int i = 0; i < 20000; ++i)
	{
		const uint32_t x = uint32_t(rng()) & 0xFFFFF, y = uint32_t(rng()) & 0xFFFFF, z = uint32_t(rng()) & 0xFFFFF;
		const uint64_t code = ref::morton(x, y, z);
		ASSERT_EQ(mo::code60(x, y, z), code);
		const auto c = mo::decode60(code);
		ASSERT_TRUE(c.x == x && c.y == y && c.z == z);
		const auto r = ref::unmorton(code);
		ASSERT_TRUE(r.x == x && r.y == y && r.z == z);
	}
}

TEST(Morton, BitPositionsAreXYZ)
{
	for (uint32_t b = 0; b < 20; ++b)
	{
		EXPECT_EQ(mo::code60(1u << b, 0, 0), uint64_t(1) << (3 * b));
		EXPECT_EQ(mo::code60(0, 1u << b, 0), uint64_t(1) << (3 * b + 1));
		EXPECT_EQ(mo::code60(0, 0, 1u << b), uint64_t(1) << (3 * b + 2));
	}
}

TEST(Morton, BitsAboveTwentyAreIgnored)
{
	std::mt19937_64 rng(11);
	for (int i = 0; i < 2000; ++i)
	{
		const uint32_t x = uint32_t(rng()), y = uint32_t(rng()), z = uint32_t(rng());
		ASSERT_EQ(mo::code60(x, y, z), mo::code60(x & 0xFFFFF, y & 0xFFFFF, z & 0xFFFFF));
		ASSERT_EQ(mo::code60(x, y, z), ref::morton(x, y, z));
	}
	// decode ignores everything above bit 59
	EXPECT_EQ(mo::decode60(~uint64_t(0)).x, (1u << 20) - 1);
	EXPECT_EQ(mo::decode60(uint64_t(0xF) << 60).z, 0u);
}

TEST(Morton, SpreadAndCompactAreInverse)
{
	for (uint32_t x = 0; x < 1024; ++x)
	{
		ASSERT_EQ(mo::compact10(mo::spread10(x)), x);
		uint32_t expected = 0;
		for (uint32_t b = 0; b < 10; ++b)
			expected |= ((x >> b) & 1u) << (3 * b);
		ASSERT_EQ(mo::spread10(x), expected);
	}
	EXPECT_EQ(mo::spread10(1024 + 5), mo::spread10(5)); // only 10 bits are used
}

TEST(Morton, PrefixNodeIndexAndCommonLevelAgainstArithmetic)
{
	std::mt19937_64 rng(3);
	for (int i = 0; i < 5000; ++i)
	{
		const uint64_t id = rng() & ((uint64_t(1) << 42) - 1);
		for (uint32_t level = 0; level <= 14; ++level)
		{
			ASSERT_EQ(mo::z_order_prefix(id, level), id / pow8(level + 1)) << id << " " << level;
			ASSERT_EQ(mo::z_order_node_id(id, level), (id / pow8(level)) % 8) << id << " " << level;
		}
		// partner that diverges at a chosen level
		const uint32_t level = uint32_t(rng() % 15);
		uint64_t other = id ^ (uint64_t(1 + rng() % 7) << (3 * level)); // differs in the digit of `level`
		other = (other & ~((uint64_t(1) << (3 * level)) - 1)) | (rng() & ((uint64_t(1) << (3 * level)) - 1)); // random below
		ASSERT_EQ(ref::common_level(id, other), level + 1) << id << " " << other;
		ASSERT_EQ(mo::common_level(id, other), level + 1) << id << " " << other;
	}
	EXPECT_EQ(mo::common_level(5, 5), 0u);
	EXPECT_EQ(mo::common_level(0, 1), 1u);
	EXPECT_EQ(mo::common_level(0, 7), 1u);
	EXPECT_EQ(mo::common_level(0, 8), 2u);
	EXPECT_EQ(mo::common_level(0, uint64_t(1) << 32), 11u);        // bit 32 is in digit 10, the paths meet above level 10
	EXPECT_EQ(mo::common_level(0, (uint64_t(1) << 42) - 1), 14u);
	EXPECT_EQ(mo::common_level(0, uint64_t(1) << 59), 20u);
}

// ----------------------------------------------------------------------------------------- getNewStatus table

TEST(OctreeReference, NewStatusTable)
{
	using namespace ref::status;
	struct Row { uint8_t child_or, child_and, level, static_dynamic; uint8_t expected; };
	const Row rows[] = {
		{ OCCUPIED, OCCUPIED, 5, 0, OCCUPIED },                        // all children occupied: pure
		{ FREE, FREE, 5, 0, FREE },                                    // all free: pure
		{ UNKNOWN, UNKNOWN, 5, 0, UNKNOWN },                           // all unknown: pure
		{ OCCUPIED | UNKNOWN, UNKNOWN, 5, 0, OCCUPIED | UNKNOWN | PART },
		{ OCCUPIED | UNKNOWN, OCCUPIED, 5, 0, OCCUPIED | UNKNOWN | PART },  // cannot happen with real children, stays PART
		{ FREE | UNKNOWN, 0, 3, 0, FREE | UNKNOWN | PART },
		{ FREE | UNKNOWN | OCCUPIED, 0, 3, 0, FREE | UNKNOWN | OCCUPIED | PART },
		{ FREE | UNKNOWN, FREE | UNKNOWN, 3, 0, FREE | UNKNOWN | PART },   // two bits in every child: not a pure state
		{ OCCUPIED, OCCUPIED, 1, STATIC, OCCUPIED | LAST_LEVEL | STATIC },
		{ OCCUPIED, OCCUPIED, 2, DYNAMIC, OCCUPIED | DYNAMIC },
		{ OCCUPIED, OCCUPIED, 7, STATIC | DYNAMIC, OCCUPIED | STATIC | DYNAMIC },
		{ OCCUPIED | UNKNOWN, 0, 1, STATIC, OCCUPIED | UNKNOWN | PART | LAST_LEVEL | STATIC },
		{ UNKNOWN, UNKNOWN, 1, 0, UNKNOWN | LAST_LEVEL },
		{ OCCUPIED, OCCUPIED, 5, COLLISION | PART | LAST_LEVEL, OCCUPIED }, // other bits of the children are not inherited
	};
	for (const Row& r : rows)
		EXPECT_EQ(ref::new_status(r.child_or, r.child_and, r.level, r.static_dynamic), r.expected)
			<< "or " << int(r.child_or) << " and " << int(r.child_and) << " level " << int(r.level);
}

// ----------------------------------------------------------------------------------------- hand derived deterministic trees

TEST(OctreeReference, SingleVoxelDeterministic)
{
	const ref::OctreeReference tree(ref::Kind::Deterministic, { { 0, 0, 0 } });
	EXPECT_EQ(tree.node_count(), 113u); // root + 14 blocks of 8
	EXPECT_EQ(tree.node_count(14), 1u);
	for (uint32_t level = 0; level < 14; ++level)
		EXPECT_EQ(tree.node_count(level), 8u) << level;
	const auto dump = tree.dump();
	EXPECT_EQ(find_node(dump, 0, 0)->status, 68);   // OCCUPIED | STATIC
	EXPECT_EQ(find_node(dump, 0, 1)->status, 66);   // UNKNOWN | STATIC
	EXPECT_EQ(find_node(dump, 0, 7)->status, 66);
	EXPECT_EQ(find_node(dump, 1, 0)->status, 94);   // OCCUPIED | UNKNOWN | PART | LAST_LEVEL | STATIC
	EXPECT_EQ(find_node(dump, 1, 5)->status, 82);   // filler: UNKNOWN | LAST_LEVEL | STATIC
	for (uint32_t level = 2; level <= 14; ++level)
		EXPECT_EQ(find_node(dump, level, 0)->status, 78) << level; // OCCUPIED | UNKNOWN | PART | STATIC
	for (uint32_t level = 2; level < 14; ++level)
		EXPECT_EQ(find_node(dump, level, 3)->status, 66) << level;
	EXPECT_EQ(tree.root().status, 78);
	EXPECT_TRUE(tree.root().has_children);
	EXPECT_FALSE(find_node(dump, 0, 0)->has_children);
	EXPECT_FALSE(find_node(dump, 1, 5)->has_children);
	EXPECT_TRUE(find_node(dump, 1, 0)->has_children);
	check_against_boxes(tree);
}

TEST(OctreeReference, FullBlocksArePureOccupied)
{
	const ref::OctreeReference two(ref::Kind::Deterministic, cube(0, 0, 0, 2));
	const auto d2 = two.dump();
	EXPECT_EQ(find_node(d2, 1, 0)->status, 84);  // OCCUPIED | LAST_LEVEL | STATIC, no PART
	EXPECT_EQ(find_node(d2, 2, 0)->status, 78);  // one occupied child, seven fillers
	EXPECT_EQ(two.node_count(), 113u);
	check_against_boxes(two);

	// 4x4x4 block at (4,4,4) = the level 2 node with id morton(1,1,1) = 7
	const ref::OctreeReference four(ref::Kind::Deterministic, cube(4, 4, 4, 4));
	const auto d4 = four.dump();
	EXPECT_EQ(find_node(d4, 2, 7)->status, 68);   // OCCUPIED | STATIC: all eight children are full
	for (uint64_t id = 56; id < 64; ++id)
		EXPECT_EQ(find_node(d4, 1, id)->status, 84) << id;
	EXPECT_EQ(find_node(d4, 3, 0)->status, 78);
	EXPECT_EQ(four.node_count(), 1u + 8u * 21u);  // inner nodes: one on each of levels 14..3 (12), the level 2 node, its 8 children
	check_against_boxes(four);
}

TEST(OctreeReference, TwoVoxelsThatMeetAtTheRoot)
{
	const ref::OctreeReference tree(ref::Kind::Deterministic, { { 0, 0, 0 }, { 8192, 0, 0 } });
	// inner nodes: the root + 13 per voxel (levels 13..1) = 27, every one owns a block of 8
	EXPECT_EQ(tree.node_count(), 1u + 27u * 8u);
	EXPECT_EQ(tree.root().status, 78);
	const auto dump = tree.dump();
	EXPECT_EQ(find_node(dump, 13, 0)->status, 78);
	EXPECT_EQ(find_node(dump, 13, 1)->status, 78);   // x is bit 0 of the root digit
	EXPECT_EQ(find_node(dump, 13, 2)->status, 66);
	EXPECT_EQ(tree.find(0).status, 68);
	EXPECT_EQ(tree.find(ref::morton(8192, 0, 0)).status, 68);
	EXPECT_EQ(tree.find(ref::morton(8192, 0, 0)).id, ref::morton(8192, 0, 0));
	check_against_boxes(tree);
}

TEST(OctreeReference, FarCornerAndFindOfUncoveredVoxels)
{
	const uint64_t last_id = (uint64_t(1) << 42) - 1;
	const ref::OctreeReference tree(ref::Kind::Deterministic, { { 16383, 16383, 16383 } });
	EXPECT_EQ(ref::morton(16383, 16383, 16383), last_id);
	EXPECT_EQ(tree.find(last_id).level, 0u);
	EXPECT_EQ(tree.find(last_id).status, 68);
	const ref::Node& far = tree.find(0);   // the other end of the grid is covered by a filler of level 13
	EXPECT_EQ(far.level, 13u);
	EXPECT_EQ(far.id, 0u);
	EXPECT_EQ(far.status, 66);
	EXPECT_FALSE(far.has_children);
	const ref::Node& sibling = tree.find(last_id - 1); // differs in the lowest digit only: a leaf filler
	EXPECT_EQ(sibling.level, 0u);
	EXPECT_EQ(sibling.id, last_id - 1);
	EXPECT_EQ(sibling.status, 66);
	const ref::Node& near = tree.find(last_id - 8);    // differs in digit 1: filler of level 1 inside the on-path level 2 node
	EXPECT_EQ(near.level, 1u);
	EXPECT_EQ(near.id, (last_id - 8) >> 3);
	EXPECT_EQ(near.status, 82);
	check_against_boxes(tree);
}

TEST(OctreeReference, EmptyDroppedAndDuplicateInput)
{
	const ref::OctreeReference empty(ref::Kind::Deterministic, {});
	EXPECT_EQ(empty.node_count(), 1u);
	EXPECT_EQ(empty.root().status, 2);
	EXPECT_FALSE(empty.root().has_children);
	EXPECT_EQ(empty.find(12345).level, 14u);
	const ref::OctreeReference empty_prob(ref::Kind::Probabilistic, {});
	EXPECT_EQ(empty_prob.root().occupancy, -128);
	EXPECT_EQ(empty_prob.root().status, 2);

	const ref::OctreeReference tree(ref::Kind::Deterministic,
		{ { 16384, 0, 0 }, { 0, 16384, 0 }, { 0, 0, 16384 }, { 5, 5, 5 }, { 5, 5, 5 }, { 1u << 20, 1, 1 } });
	EXPECT_EQ(tree.dropped(), 4u);
	EXPECT_EQ(tree.voxel_count(), 1u);
	EXPECT_EQ(tree.node_count(), 113u);
}

TEST(OctreeReference, DumpIsOrderedAndUnique)
{
	std::mt19937_64 rng(5);
	const ref::OctreeReference tree(ref::Kind::Deterministic, random_voxels(rng, 300, 16384));
	const auto dump = tree.dump();
	for (size_t i = 1; i < dump.size(); ++i)
		ASSERT_TRUE(std::make_pair(dump[i - 1].level, dump[i - 1].id) < std::make_pair(dump[i].level, dump[i].id)) << i;
	size_t sum = 0;
	for (uint32_t level = 0; level <= 14; ++level)
		sum += tree.node_count(level);
	EXPECT_EQ(sum, tree.node_count());
}

// ------------------------------------------------------------------------------------ hand derived probabilistic trees

TEST(OctreeReference, SingleVoxelProbabilistic)
{
	const ref::OctreeReference tree(ref::Kind::Probabilistic, { { 0, 0, 0 } });
	EXPECT_EQ(tree.node_count(), 113u);
	const auto dump = tree.dump();
	const auto* leaf = find_node(dump, 0, 0);
	EXPECT_EQ(leaf->occupancy, 127);
	EXPECT_EQ(leaf->status, 64);
	const auto* filler = find_node(dump, 0, 1);
	EXPECT_EQ(filler->occupancy, -128);
	EXPECT_EQ(filler->status, 66);
	const auto* l1 = find_node(dump, 1, 0);
	EXPECT_EQ(l1->occupancy, 127);
	EXPECT_EQ(l1->status, 90);   // UNKNOWN | PART | LAST_LEVEL | STATIC
	EXPECT_EQ(find_node(dump, 1, 6)->status, 82);
	EXPECT_EQ(find_node(dump, 1, 6)->occupancy, -128);
	for (uint32_t level = 2; level <= 14; ++level)
	{
		EXPECT_EQ(find_node(dump, level, 0)->status, 74) << level; // UNKNOWN | PART | STATIC
		EXPECT_EQ(find_node(dump, level, 0)->occupancy, 127) << level;
	}
	EXPECT_EQ(tree.root().occupancy, 127);
	check_against_boxes(tree);
}

TEST(OctreeReference, FullBlockIsNotPartInTheProbabilisticTree)
{
	const ref::OctreeReference tree(ref::Kind::Probabilistic, cube(0, 0, 0, 2));
	const auto dump = tree.dump();
	EXPECT_EQ(find_node(dump, 1, 0)->status, 82);   // UNKNOWN | LAST_LEVEL | STATIC: all children 127
	EXPECT_EQ(find_node(dump, 1, 0)->occupancy, 127);
	EXPECT_EQ(find_node(dump, 2, 0)->status, 74);   // children differ (127 / -128)
	check_against_boxes(tree);

	// all eight children partial but equal occupancy: still PART because a child is PART
	std::vector<ref::Voxel> v;
	for (uint32_t c = 0; c < 8; ++c)
		v.push_back({ (c & 1u) * 2u, ((c >> 1) & 1u) * 2u, ((c >> 2) & 1u) * 2u }); // one voxel in each level 1 node of a level 2 node
	const ref::OctreeReference partial(ref::Kind::Probabilistic, v);
	const auto d = partial.dump();
	EXPECT_EQ(find_node(d, 2, 0)->occupancy, 127);
	EXPECT_EQ(find_node(d, 2, 0)->status, 74);
	check_against_boxes(partial);
}

// ------------------------------------------------------------------------------------------------ brute force oracle

TEST(OctreeReference, RandomSetsMatchTheBoxOracle)
{
	std::mt19937_64 rng(2024);
	for (const auto kind : { ref::Kind::Deterministic, ref::Kind::Probabilistic })
	{
		for (size_t count : { size_t(1), size_t(2), size_t(10), size_t(200), size_t(3000) })
		{
			check_against_boxes(ref::OctreeReference(kind, random_voxels(rng, count, 16384)));   // sparse over the whole grid
			check_against_boxes(ref::OctreeReference(kind, random_voxels(rng, count, 16, 100, 200, 300))); // dense cluster
			check_against_boxes(ref::OctreeReference(kind, random_voxels(rng, count, 8, 16376, 16376, 16376))); // at the far corner
		}
		for (uint32_t side : { 1u, 2u, 3u, 4u, 5u, 8u, 9u })
			check_against_boxes(ref::OctreeReference(kind, cube(side * 3, side, 2 * side, side)));
		// a full cube with random holes
		auto holes = cube(32, 32, 32, 8);
		for (int i = 0; i < 40; ++i)
			holes.erase(holes.begin() + long(rng() % holes.size()));
		check_against_boxes(ref::OctreeReference(kind, holes));
	}
}

TEST(OctreeReference, FindCoversEveryVoxelOfASmallGrid)
{
	std::mt19937_64 rng(77);
	const auto voxels = random_voxels(rng, 60, 8);
	for (const auto kind : { ref::Kind::Deterministic, ref::Kind::Probabilistic })
	{
		const ref::OctreeReference tree(kind, voxels);
		std::set<uint64_t> occupied;
		for (const auto& v : voxels)
			occupied.insert(ref::morton(v.x, v.y, v.z));
		for (uint64_t id = 0; id < 512; ++id)
		{
			const ref::Node& n = tree.find(id);
			ASSERT_FALSE(n.has_children);
			ASSERT_EQ(n.id, id >> (3 * n.level));
			// the walk stops at the highest node on the path whose box holds no occupied voxel, or at the voxel itself
			uint32_t expected_level = 0;
			for (int level = 14; level > 0; --level)
				if (voxels_in_box(tree, uint32_t(level), id >> (3 * level)) == 0)
				{
					expected_level = uint32_t(level);
					break;
				}
			ASSERT_EQ(n.level, expected_level) << id;
			if (expected_level == 0)
				ASSERT_EQ(occupied.count(id) == 1, kind == ref::Kind::Deterministic ? (n.status & ref::status::OCCUPIED) != 0 : n.occupancy == ref::kOccupied) << id;
		}
	}
}

// ---------------------------------------------------------------------------------------------------- node formats

namespace
{
	constexpr OctreeKind kKinds[] = { OctreeKind::Deterministic, OctreeKind::Probabilistic };

	uint64_t random_pointer(std::mt19937_64& rng)
	{
		const uint64_t low = rng() & ((uint64_t(1) << 47) - 1);
		return (rng() & 1) ? (low | 0xFFFF800000000000ull) : low; // canonical: bit 47 copied into the upper 16 bits
	}
}

TEST(OctreeFormat, WordsOfHandBuiltNodes)
{
	std::vector<uint8_t> block(64, 0);
	fmt::HostNode det;
	det.status = 0x44;
	det.flags = 0x80;
	det.child = 0x0000123456789ABCull;
	fmt::encode_inner(OctreeKind::Deterministic, block.data(), 3, det);
	EXPECT_EQ(fmt::word(block.data(), 6), 0x9ABC8044u);
	EXPECT_EQ(fmt::word(block.data(), 7), 0x12345678u);
	EXPECT_EQ(fmt::word(block.data(), 5), 0u); // neighbours untouched
	EXPECT_EQ(fmt::word(block.data(), 8), 0u);

	std::vector<uint8_t> prob_block(96, 0);
	fmt::HostNode prob;
	prob.status = 0x4A;
	prob.flags = 0x01;
	prob.occupancy = -1;
	prob.child = 0xFFFF800000001000ull;
	fmt::encode_inner(OctreeKind::Probabilistic, prob_block.data(), 2, prob);
	EXPECT_EQ(fmt::word(prob_block.data(), 6), 0x00FF014Au);
	EXPECT_EQ(fmt::word(prob_block.data(), 7), 0x00001000u);
	EXPECT_EQ(fmt::word(prob_block.data(), 8), 0x00008000u); // only 48 bits are stored
	const auto back = fmt::decode_inner(OctreeKind::Probabilistic, prob_block.data(), 2);
	EXPECT_EQ(back.child, 0xFFFF800000001000ull);
	EXPECT_EQ(back.occupancy, -1);

	std::vector<uint8_t> leaves(8, 0);
	for (uint32_t s = 0; s < 8; ++s)
	{
		fmt::HostNode leaf;
		leaf.status = uint8_t(0x10 + s);
		fmt::encode_leaf(OctreeKind::Deterministic, leaves.data(), s, leaf);
	}
	EXPECT_EQ(fmt::word(leaves.data(), 0), 0x13121110u);
	EXPECT_EQ(fmt::word(leaves.data(), 1), 0x17161514u);

	std::vector<uint8_t> prob_leaves(16, 0);
	for (uint32_t s = 0; s < 8; ++s)
	{
		fmt::HostNode leaf;
		leaf.status = uint8_t(0x10 + s);
		leaf.occupancy = int8_t(s == 3 ? -128 : 100 + s);
		fmt::encode_leaf(OctreeKind::Probabilistic, prob_leaves.data(), s, leaf);
	}
	EXPECT_EQ(fmt::word(prob_leaves.data(), 0), 0x65116410u);
	EXPECT_EQ(fmt::word(prob_leaves.data(), 1), 0x80136612u);
	EXPECT_EQ(fmt::word(prob_leaves.data(), 2), 0x69156814u);
	EXPECT_EQ(fmt::word(prob_leaves.data(), 3), 0x6B176A16u);
	EXPECT_EQ(fmt::decode_leaf(OctreeKind::Probabilistic, prob_leaves.data(), 3).occupancy, -128);
	EXPECT_EQ(fmt::decode_leaf(OctreeKind::Probabilistic, prob_leaves.data(), 6).status, 0x16);
	EXPECT_EQ(fmt::decode_leaf(OctreeKind::Deterministic, leaves.data(), 6).status, 0x16);
}

TEST(OctreeFormat, NodesSurviveEncodeDecodeAndLeaveTheirNeighboursAlone)
{
	std::mt19937_64 rng(31);
	for (const auto kind : kKinds)
		for (int i = 0; i < 3000; ++i)
		{
			std::vector<uint8_t> block(fmt::inner_block_bytes(kind));
			for (auto& b : block)
				b = uint8_t(rng());
			const auto before = block;
			fmt::HostNode n;
			n.status = uint8_t(rng());
			n.flags = uint8_t(rng());
			n.occupancy = kind == OctreeKind::Probabilistic ? int8_t(rng()) : int8_t(0);
			n.child = (rng() % 4 == 0) ? 0 : random_pointer(rng);
			const uint32_t slot = uint32_t(rng() % 8);
			fmt::encode_inner(kind, block.data(), slot, n);
			const auto back = fmt::decode_inner(kind, block.data(), slot);
			ASSERT_TRUE(back.status == n.status && back.flags == n.flags && back.occupancy == n.occupancy && back.child == n.child) << i;
			const size_t node_bytes = fmt::inner_block_bytes(kind) / 8;
			for (size_t b = 0; b < block.size(); ++b)
				if (b / node_bytes != slot)
					ASSERT_EQ(block[b], before[b]) << "byte " << b << " of another node changed";

			std::vector<uint8_t> leaves(fmt::leaf_block_bytes(kind));
			for (auto& b : leaves)
				b = uint8_t(rng());
			const auto leaves_before = leaves;
			fmt::encode_leaf(kind, leaves.data(), slot, n);
			const auto leaf = fmt::decode_leaf(kind, leaves.data(), slot);
			ASSERT_TRUE(leaf.status == n.status && leaf.occupancy == n.occupancy) << i;
			const size_t leaf_bytes = fmt::leaf_block_bytes(kind) / 8;
			for (size_t b = 0; b < leaves.size(); ++b)
				if (b / leaf_bytes != slot)
					ASSERT_EQ(leaves[b], leaves_before[b]);
		}
}

TEST(OctreeFormat, PointersAreCanonical48Bit)
{
	EXPECT_EQ(fmt::sign_extend_48(0x0000123456789ABCull), 0x0000123456789ABCull);
	EXPECT_EQ(fmt::sign_extend_48(0x0000800000000000ull), 0xFFFF800000000000ull);
	EXPECT_EQ(fmt::sign_extend_48(0xFFFF800000001000ull), 0xFFFF800000001000ull);
	EXPECT_EQ(fmt::sign_extend_48(0x00007FFFFFFFFFFFull), 0x00007FFFFFFFFFFFull);
	EXPECT_EQ(fmt::sign_extend_48(0), 0u);
	EXPECT_TRUE(fmt::addressable(0x1000, 64));
	EXPECT_FALSE(fmt::addressable(0x1000, 0));
	EXPECT_TRUE(fmt::addressable(0xFFFF800000001000ull, 4096));
	EXPECT_FALSE(fmt::addressable(0x0001000000000000ull, 64));  // bit 48 set, bit 47 clear
	EXPECT_FALSE(fmt::addressable(0xFFFF000000000000ull, 64));  // bit 47 clear, upper bits set
	EXPECT_FALSE(fmt::addressable(0x00007FFFFFFFFFF0ull, 64));  // crosses into the upper half
	EXPECT_TRUE(fmt::addressable(0x00007FFFFFFFFFC0ull, 64));   // ends exactly at the boundary
	EXPECT_FALSE(fmt::addressable(0xFFFFFFFFFFFFFFF0ull, 64));  // wraps around
}

TEST(OctreeFormat, ParentRuleMatchesTheReference)
{
	std::mt19937_64 rng(32);
	for (int i = 0; i < 20000; ++i)
	{
		const uint32_t level = uint32_t(1 + rng() % 14);
		// deterministic: children with plausible statuses (occupancy bits 1..7, other bits random)
		fmt::HostNode kids[8];
		uint8_t o = 0, a = 0xFF, sd = 0;
		const bool uniform = rng() % 3 == 0;
		const uint8_t base = uint8_t(1u << (rng() % 3));
		for (auto& k : kids)
		{
			const uint8_t occ = uniform ? base : uint8_t(1 + rng() % 7);
			k.status = uint8_t(occ | (rng() & (ref::status::PART | ref::status::LAST_LEVEL | ref::status::COLLISION | ref::status::STATIC | ref::status::DYNAMIC)));
			o |= k.status & fmt::kOccupancyMask;
			a &= k.status & fmt::kOccupancyMask;
			sd |= k.status;
		}
		ASSERT_EQ(fmt::parent_of(OctreeKind::Deterministic, kids, level).status, ref::new_status(o, a, level, sd)) << i;

		// probabilistic: occupancy = max, PART unless all equal and no child is PART
		fmt::HostNode pk[8];
		int pmax = -128, pmin = 127;
		bool any_part = false;
		const bool same = rng() % 3 == 0;
		const int8_t common = int8_t(rng());
		for (auto& k : pk)
		{
			k.occupancy = same ? common : int8_t(rng());
			k.status = uint8_t(ref::status::UNKNOWN | ref::status::STATIC | ((rng() % 4 == 0) ? ref::status::PART : 0));
			pmax = std::max<int>(pmax, k.occupancy);
			pmin = std::min<int>(pmin, k.occupancy);
			any_part = any_part || (k.status & ref::status::PART);
		}
		const auto parent = fmt::parent_of(OctreeKind::Probabilistic, pk, level);
		ASSERT_EQ(parent.occupancy, pmax) << i;
		const uint8_t expected = uint8_t(ref::status::UNKNOWN | ref::status::STATIC | ((pmax != pmin || any_part) ? ref::status::PART : 0) | (level == 1 ? ref::status::LAST_LEVEL : 0));
		ASSERT_EQ(parent.status, expected) << i;
	}
}

TEST(OctreeFormat, FillerAndBlockSizes)
{
	EXPECT_EQ(fmt::filler(OctreeKind::Deterministic, 5).status, 66);
	EXPECT_EQ(fmt::filler(OctreeKind::Deterministic, 1).status, 82);
	EXPECT_EQ(fmt::filler(OctreeKind::Deterministic, 0).status, 66);
	EXPECT_EQ(fmt::filler(OctreeKind::Probabilistic, 5).occupancy, -128);
	EXPECT_EQ(fmt::filler(OctreeKind::Deterministic, 5).occupancy, 0);
	EXPECT_EQ(fmt::block_bytes(OctreeKind::Deterministic, 0), 8u);
	EXPECT_EQ(fmt::block_bytes(OctreeKind::Deterministic, 1), 64u);
	EXPECT_EQ(fmt::block_bytes(OctreeKind::Probabilistic, 0), 16u);
	EXPECT_EQ(fmt::block_bytes(OctreeKind::Probabilistic, 14), 96u);
}
