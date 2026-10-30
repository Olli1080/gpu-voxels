// Slice S6c (second part): collision of two octrees by simultaneous traversal.
// Expectations: the independent recursion in tests/reference/octree_reference.h (collide_tree) and, for exact collisions, the
// plain intersection of the two voxel sets.
#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
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
	ref::Kind to_ref(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? ref::Kind::Probabilistic : ref::Kind::Deterministic; }
	const char* name(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? "probabilistic" : "deterministic"; }

	std::vector<ref::Voxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent, uint32_t ox = 0, uint32_t oy = 0, uint32_t oz = 0)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<ref::Voxel> v;
		for (size_t i = 0; i < count; ++i)
			v.push_back({ ox + d(rng), oy + d(rng), oz + d(rng) });
		return v;
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

	std::vector<OctreeVoxel> to_voxels(const std::vector<ref::Voxel>& v)
	{
		std::vector<OctreeVoxel> out;
		for (const auto& x : v)
			out.push_back({ x.x, x.y, x.z });
		return out;
	}

	uint64_t intersection_size(const std::vector<ref::Voxel>& a, const std::vector<ref::Voxel>& b)
	{
		std::set<uint64_t> in_a;
		for (const auto& v : a)
			in_a.insert(ref::morton(v.x, v.y, v.z));
		std::set<uint64_t> in_both;
		for (const auto& v : b)
			if (in_a.count(ref::morton(v.x, v.y, v.z)))
				in_both.insert(ref::morton(v.x, v.y, v.z));
		return in_both.size();
	}

	void expect_dump(const Octree& tree, const std::vector<ref::Node>& expected)
	{
		const auto dump = tree.dump();
		ASSERT_EQ(dump.size(), expected.size());
		for (size_t i = 0; i < dump.size(); ++i)
			ASSERT_TRUE(dump[i].level == expected[i].level && dump[i].id == expected[i].id && dump[i].status == expected[i].status &&
				dump[i].occupancy == expected[i].occupancy && dump[i].has_children == expected[i].has_children)
				<< "node " << i << " level " << dump[i].level << " id " << dump[i].id << " status " << int(dump[i].status) << " expected " << int(expected[i].status);
	}

	/// Collides the trees built from `a` and `b` with all option variants and compares everything with the reference.
	void verify_pair(Context& ctx, OctreeKind kind, const std::vector<ref::Voxel>& a, const std::vector<ref::Voxel>& b, const std::string& label,
		bool exact_oracle = true)
	{
		SCOPED_TRACE(std::string(name(kind)) + ": " + label);
		const ref::OctreeReference ref_a(to_ref(kind), a), ref_b(to_ref(kind), b);
		Octree tree_a(ctx, kind), tree_b(ctx, kind);
		tree_a.build(to_voxels(a));
		tree_b.build(to_voxels(b));
		for (uint32_t min_level : { 0u, 1u, 2u, 4u, 13u, 14u })
			for (const bool mark : { true, false })
			{
				OctreeTreeCollisionOptions options;
				options.min_level = min_level;
				options.mark_collisions = mark;
				const uint64_t volume = tree_a.collide_with(tree_b, options);
				const auto expected = ref_a.collide_tree(ref_b, min_level, 10, mark);
				ASSERT_EQ(volume, expected.voxels) << "min_level " << min_level << " mark " << mark;
				if (min_level == 0 && exact_oracle)
					ASSERT_EQ(volume, intersection_size(a, b)) << "the exact collision is the intersection of the voxel sets";

				ASSERT_NO_THROW(tree_a.check_tree());
				ASSERT_NO_THROW(tree_b.check_tree());
				expect_dump(tree_a, ref_a.dump_with_collisions(expected.flagged_this));
				expect_dump(tree_b, ref_b.dump_with_collisions(expected.flagged_other));
				if (mark)
				{
					tree_a.clear_collision_flags();
					tree_b.clear_collision_flags();
					expect_dump(tree_a, ref_a.dump());
					expect_dump(tree_b, ref_b.dump());
				}
			}
		// extreme occupancy thresholds of the probabilistic collider: -128 would make unknown nodes occupied if the unknown test were missing
		for (const int32_t threshold : { -128, -1, 127 })
			for (const uint32_t min_level : { 0u, 3u })
			{
				OctreeTreeCollisionOptions options;
				options.min_level = min_level;
				options.mark_collisions = false;
				options.occupancy_threshold = threshold;
				ASSERT_EQ(tree_a.collide_with(tree_b, options), ref_a.collide_tree(ref_b, min_level, threshold, false).voxels)
					<< "threshold " << threshold << " min_level " << min_level;
			}
	}
}

TEST_F(VulkanTest, OctreeTreeCollisionMatchesTheReferenceOnHandMadeSets)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		verify_pair(ctx(), kind, { { 0, 0, 0 } }, { { 0, 0, 0 } }, "same voxel");
		verify_pair(ctx(), kind, { { 0, 0, 0 } }, { { 1, 0, 0 } }, "neighbours");
		verify_pair(ctx(), kind, { { 0, 0, 0 } }, { { 16383, 16383, 16383 } }, "far apart");
		verify_pair(ctx(), kind, cube(0, 0, 0, 8), cube(4, 4, 4, 8), "overlapping cubes (uniform nodes with kept children)");
		verify_pair(ctx(), kind, cube(0, 0, 0, 8), { { 3, 3, 3 }, { 9, 9, 9 } }, "cube against single voxels");
		verify_pair(ctx(), kind, cube(0, 0, 0, 2), cube(0, 0, 0, 2), "equal 2x2x2 cubes");
		verify_pair(ctx(), kind, cube(0, 0, 0, 4), cube(2, 0, 0, 4), "4 cubes shifted by 2");
		verify_pair(ctx(), kind, { { 0, 0, 0 }, { 8192, 0, 0 } }, { { 8192, 0, 0 }, { 0, 8192, 0 } }, "voxels meeting at the root");
		verify_pair(ctx(), kind, {}, cube(0, 0, 0, 4), "empty tree against a cube");
		verify_pair(ctx(), kind, {}, {}, "two empty trees");
	}
}

TEST_F(VulkanTest, OctreeTreeCollisionMatchesTheReferenceOnRandomSets)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(21);
	for (const auto kind : kKinds)
	{
		verify_pair(ctx(), kind, random_voxels(rng, 400, 24, 100, 100, 100), random_voxels(rng, 400, 24, 108, 108, 108), "dense clusters");
		verify_pair(ctx(), kind, random_voxels(rng, 300, 16384), random_voxels(rng, 300, 16384), "sparse over the grid (few collisions)");
		auto shared = random_voxels(rng, 200, 16384);
		auto a = random_voxels(rng, 200, 16384);
		auto b = random_voxels(rng, 200, 16384);
		a.insert(a.end(), shared.begin(), shared.end());
		b.insert(b.end(), shared.begin(), shared.end());
		verify_pair(ctx(), kind, a, b, "sparse sets with 200 common voxels");
		auto holes = cube(32, 32, 32, 8);
		for (int i = 0; i < 60; ++i)
			holes.erase(holes.begin() + long(rng() % holes.size()));
		verify_pair(ctx(), kind, holes, random_voxels(rng, 500, 14, 30, 30, 30), "cube with holes against a random cluster");
		verify_pair(ctx(), kind, random_voxels(rng, 1500, 40, 0, 0, 0), random_voxels(rng, 1500, 40, 20, 20, 20), "large dense overlap");
	}
}

TEST_F(VulkanTest, OctreeTreeCollisionOfIdenticalAndDisjointTrees)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(22);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto voxels = random_voxels(rng, 800, 30, 500, 500, 500);
		Octree a(ctx(), kind), b(ctx(), kind), empty(ctx(), kind);
		a.build(to_voxels(voxels));
		b.build(to_voxels(voxels));
		OctreeTreeCollisionOptions options;
		options.mark_collisions = false;
		EXPECT_EQ(a.collide_with(b, options), a.voxel_count());
		EXPECT_EQ(a.collide_with(empty, options), 0u);
		EXPECT_EQ(empty.collide_with(a, options), 0u);
		Octree far_away(ctx(), kind);
		far_away.build(to_voxels(random_voxels(rng, 800, 30, 9000, 9000, 9000)));
		EXPECT_EQ(a.collide_with(far_away, options), 0u);
		// a coarser lookup is conservative: never less volume than the exact one, and boxes that are far apart still do not collide
		options.min_level = 3;
		EXPECT_EQ(a.collide_with(far_away, options), 0u);
		EXPECT_GE(a.collide_with(b, options), a.voxel_count());
		options.min_level = 14;
		EXPECT_EQ(a.collide_with(b, options), uint64_t(1) << 39) << "on the level of the root's children one pair of nodes of 8^13 voxels collides";
	}
}

TEST_F(VulkanTest, OctreeTreeCollisionWithUniformRoots)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	// A root without children that is occupied is a uniform node covering the whole grid. Built trees never have one (it needs 2^42
	// voxels), so it is forged by overwriting the root node; the expectation is the plain voxel set of the other tree.
	std::mt19937_64 rng(23);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		auto forge_uniform_root = [&](Octree& tree)
		{
			std::vector<uint8_t> block(fmt::inner_block_bytes(kind));
			tree.chunk(14).download(block.data(), block.size());
			fmt::HostNode root = fmt::decode_inner(kind, block.data(), 0);
			root.status = uint8_t(fmt::kOccupied | fmt::kStatic);
			root.child = 0;
			root.occupancy = kind == OctreeKind::Probabilistic ? fmt::kProbMax : int8_t(0);
			fmt::encode_inner(kind, block.data(), 0, root);
			tree.chunk(14).upload(block.data(), block.size());
		};
		const auto voxels = random_voxels(rng, 500, 40, 700, 800, 900);
		Octree uniform(ctx(), kind), other(ctx(), kind);
		uniform.build(std::vector<OctreeVoxel>{ { 1, 2, 3 } });
		forge_uniform_root(uniform);
		other.build(to_voxels(voxels));
		OctreeTreeCollisionOptions options;
		options.mark_collisions = false;
		const ref::OctreeReference expected_other(to_ref(kind), voxels);
		EXPECT_EQ(uniform.collide_with(other, options), expected_other.voxel_count()) << "the uniform root stands in for every child of the other tree";
		EXPECT_EQ(other.collide_with(uniform, options), expected_other.voxel_count());

		Octree uniform2(ctx(), kind);
		uniform2.build(std::vector<OctreeVoxel>{ { 5, 5, 5 } });
		forge_uniform_root(uniform2);
		EXPECT_EQ(uniform.collide_with(uniform2, options), uint64_t(1) << 42) << "two uniform roots: the whole grid";
		options.mark_collisions = true;
		Octree uniform3(ctx(), kind), uniform4(ctx(), kind);
		uniform3.build(std::vector<OctreeVoxel>{ { 5, 5, 5 } });
		uniform4.build(std::vector<OctreeVoxel>{ { 6, 6, 6 } });
		forge_uniform_root(uniform3);
		forge_uniform_root(uniform4);
		EXPECT_EQ(uniform3.collide_with(uniform4, options), uint64_t(1) << 42);
		std::vector<uint8_t> block(fmt::inner_block_bytes(kind));
		uniform3.chunk(14).download(block.data(), block.size());
		EXPECT_TRUE(fmt::decode_inner(kind, block.data(), 0).status & fmt::kCollision) << "the first tree's root is marked";
		uniform4.chunk(14).download(block.data(), block.size());
		EXPECT_FALSE(fmt::decode_inner(kind, block.data(), 0).status & fmt::kCollision);
	}
}

TEST_F(VulkanTest, OctreeTreeCollisionTreatsAMissingChildPointerAsUniform)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(26);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		// a root with the PART flag but without a child block is corrupt; the kernels must not follow the null pointer (that would lose
		// the device) but handle the node like a uniform one
		Octree forged(ctx(), kind), other(ctx(), kind);
		forged.build(std::vector<OctreeVoxel>{ { 1, 2, 3 } });
		std::vector<uint8_t> block(fmt::inner_block_bytes(kind));
		forged.chunk(14).download(block.data(), block.size());
		fmt::HostNode root = fmt::decode_inner(kind, block.data(), 0);
		root.status = uint8_t(fmt::kOccupied | fmt::kStatic | fmt::kPart);
		root.child = 0;
		root.occupancy = kind == OctreeKind::Probabilistic ? fmt::kProbMax : int8_t(0);
		fmt::encode_inner(kind, block.data(), 0, root);
		forged.chunk(14).upload(block.data(), block.size());

		const auto voxels = random_voxels(rng, 300, 30, 600, 700, 800);
		other.build(to_voxels(voxels));
		OctreeTreeCollisionOptions options;
		options.mark_collisions = false;
		const uint64_t expected = ref::OctreeReference(to_ref(kind), voxels).voxel_count();
		EXPECT_EQ(forged.collide_with(other, options), expected);
		EXPECT_EQ(other.collide_with(forged, options), expected);
	}
}

TEST_F(VulkanTest, OctreeTreeCollisionUsesTheOccupancyThresholdOfTheLeaves)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	// Built trees only know the occupancies 127 and -128, so the leaves are overwritten with other values. The inner nodes keep 127
	// (occupied for every threshold below), sparse sets avoid full blocks whose leaves would be ignored.
	std::mt19937_64 rng(25);
	const auto va = random_voxels(rng, 220, 12, 100, 100, 100);
	const auto vb = random_voxels(rng, 220, 12, 102, 102, 102);
	Octree a(ctx(), OctreeKind::Probabilistic), b(ctx(), OctreeKind::Probabilistic);
	a.build(to_voxels(va));
	b.build(to_voxels(vb));
	const int8_t values[] = { -128, -127, -1, 0, 9, 10, 11, 59, 60, 61, 126, 127 };
	auto forge = [&](Octree& tree)
	{
		std::vector<uint8_t> bytes(tree.chunk(0).size());
		tree.chunk(0).download(bytes.data(), bytes.size());
		for (size_t offset = 0; offset < bytes.size(); offset += fmt::leaf_block_bytes(OctreeKind::Probabilistic))
			for (uint32_t slot = 0; slot < 8; ++slot)
			{
				auto leaf = fmt::decode_leaf(OctreeKind::Probabilistic, bytes.data() + offset, slot);
				leaf.occupancy = values[rng() % std::size(values)];
				fmt::encode_leaf(OctreeKind::Probabilistic, bytes.data() + offset, slot, leaf);
			}
		tree.chunk(0).upload(bytes.data(), bytes.size());
	};
	forge(a);
	forge(b);
	auto leaves_of = [](const Octree& tree)
	{
		std::map<uint64_t, int> leaves;
		for (const auto& n : tree.dump())
			if (n.level == 0)
				leaves[n.id] = n.occupancy;
		return leaves;
	};
	const auto la = leaves_of(a), lb = leaves_of(b);
	for (const int32_t threshold : { 10, -128, -127, 0, 11, 60, 127 })
	{
		uint64_t expected = 0;
		for (const auto& [id, occupancy] : la)
		{
			const auto it = lb.find(id);
			if (it != lb.end() && occupancy != -128 && occupancy >= threshold && it->second != -128 && it->second >= threshold)
				++expected;
		}
		OctreeTreeCollisionOptions options;
		options.mark_collisions = false;
		options.occupancy_threshold = threshold;
		EXPECT_EQ(a.collide_with(b, options), expected) << "threshold " << threshold;
		EXPECT_EQ(b.collide_with(a, options), expected) << "threshold " << threshold;
	}
}

TEST_F(VulkanTest, OctreeTreeCollisionRejectsInvalidPartners)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree det(ctx(), OctreeKind::Deterministic), prob(ctx(), OctreeKind::Probabilistic);
	EXPECT_THROW((void)det.collide_with(prob), VulkanError);
	EXPECT_THROW((void)det.collide_with(det), VulkanError);
}

TEST_F(VulkanTest, OctreeTreeCollisionOfLargeTrees)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(24);
	const auto va = random_voxels(rng, 1'000'000, 200, 5000, 6000, 7000);
	const auto vb = random_voxels(rng, 1'000'000, 200, 5100, 6100, 7100);
	const uint64_t expected = intersection_size(va, vb);
	for (const auto kind : kKinds)
	{
		Octree a(ctx(), kind), b(ctx(), kind);
		a.build(to_voxels(va));
		b.build(to_voxels(vb));
		OctreeTreeCollisionOptions options;
		const auto t0 = std::chrono::steady_clock::now();
		const uint64_t volume = a.collide_with(b, options);
		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "[vulkan] octree vs octree (" << name(kind) << "), 2 x ~1 M voxels: " << volume << " voxels in collision in "
			<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms (" << ctx().info().name << ")" << std::endl;
		EXPECT_EQ(volume, expected);
		EXPECT_GT(volume, 0u);
		EXPECT_NO_THROW(a.check_tree());
		EXPECT_NO_THROW(b.check_tree());
	}
}
