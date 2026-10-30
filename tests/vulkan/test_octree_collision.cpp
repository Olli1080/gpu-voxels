// Slice S6c (first part): collisions of the octree with bit maps, probabilistic maps, bit lists and Morton ids, collision flags.
// The expectations come from the independent octree reference (tests/reference/octree_reference.h) and from the downloaded
// contents of the maps / lists, so the tree code is not compared with itself.
#include "vulkan_fixture.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>

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

	struct Probes
	{
		std::vector<uint64_t> ids;
		std::vector<std::array<uint32_t, 8>> flags;
	};

	/// Probe at coordinates + offset, or nothing when the result lies outside the 2^14 grid.
	void add_probe(Probes& probes, int64_t x, int64_t y, int64_t z, const std::array<uint32_t, 8>& flags)
	{
		if (x < 0 || y < 0 || z < 0 || x >= 16384 || y >= 16384 || z >= 16384)
			return;
		probes.ids.push_back(ref::morton(uint32_t(x), uint32_t(y), uint32_t(z)));
		probes.flags.push_back(flags);
	}

	Probes probes_of_bit_words(const std::vector<uint32_t>& words, Dim3ui dims, const OctreeCollisionOptions& o)
	{
		Probes probes;
		for (uint64_t i = 0; i < dims.voxel_count(); ++i)
		{
			std::array<uint32_t, 8> f{};
			uint32_t any = 0;
			for (size_t w = 0; w < 8; ++w)
				any |= f[w] = words[8 * i + w];
			if (!any)
				continue;
			add_probe(probes, int64_t(i % dims.x) + o.offset_x, int64_t((i / dims.x) % dims.y) + o.offset_y, int64_t(i / (uint64_t(dims.x) * dims.y)) + o.offset_z, f);
		}
		return probes;
	}

	Probes probes_of_prob(const std::vector<int8_t>& values, Dim3ui dims, const OctreeCollisionOptions& o)
	{
		Probes probes;
		for (uint64_t i = 0; i < dims.voxel_count(); ++i)
			if (values[i] >= o.map_occupancy_threshold)
				add_probe(probes, int64_t(i % dims.x) + o.offset_x, int64_t((i / dims.x) % dims.y) + o.offset_y, int64_t(i / (uint64_t(dims.x) * dims.y)) + o.offset_z, {});
		return probes;
	}

	std::vector<ref::Voxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent, uint32_t ox = 0, uint32_t oy = 0, uint32_t oz = 0)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<ref::Voxel> v;
		for (size_t i = 0; i < count; ++i)
			v.push_back({ ox + d(rng), oy + d(rng), oz + d(rng) });
		return v;
	}

	std::vector<OctreeVoxel> to_voxels(const std::vector<ref::Voxel>& v)
	{
		std::vector<OctreeVoxel> out;
		for (const auto& x : v)
			out.push_back({ x.x, x.y, x.z });
		return out;
	}

	std::vector<Coord3ui> random_coords(std::mt19937_64& rng, size_t count, Dim3ui dims)
	{
		std::vector<Coord3ui> c;
		for (size_t i = 0; i < count; ++i)
			c.push_back({ uint32_t(rng() % dims.x), uint32_t(rng() % dims.y), uint32_t(rng() % dims.z) });
		return c;
	}

	bool same_node(const OctreeNode& a, const ref::Node& b)
	{
		return a.level == b.level && a.id == b.id && a.status == b.status && a.occupancy == b.occupancy && a.has_children == b.has_children;
	}

	void expect_tree_equals(const Octree& tree, const std::vector<ref::Node>& expected)
	{
		const auto dump = tree.dump();
		ASSERT_EQ(dump.size(), expected.size());
		for (size_t i = 0; i < dump.size(); ++i)
			ASSERT_TRUE(same_node(dump[i], expected[i])) << "node " << i << " level " << dump[i].level << " id " << dump[i].id << " status " << int(dump[i].status)
				<< " expected " << int(expected[i].status);
	}

	/// Runs one collision query and checks result, marked nodes, check_tree and clearing.
	template<typename Query>
	void check_query(Octree& tree, const ref::OctreeReference& expected_tree, const Probes& probes, bool probes_have_flags, const OctreeCollisionOptions& options,
		Query&& query, const std::string& label)
	{
		SCOPED_TRACE(label + " min_level " + std::to_string(options.min_level) + " offset " + std::to_string(options.offset_x) + "," + std::to_string(options.offset_y) + "," +
			std::to_string(options.offset_z));
		const auto expected = expected_tree.collide(probes.ids, probes_have_flags ? &probes.flags : nullptr, options.min_level, options.tree_occupancy_threshold);
		const OctreeCollisionResult result = query(tree, options);
		ASSERT_EQ(result.collisions, expected.collisions);
		if (options.count_unknown)
			ASSERT_EQ(result.collisions_with_unknown, expected.with_unknown);
		else
			ASSERT_EQ(result.collisions_with_unknown, 0u);
		if (options.compute_voxel_flags && probes_have_flags)
			ASSERT_EQ(result.voxel_flags, BitVector256(expected.flags));
		else
			ASSERT_EQ(result.voxel_flags, BitVector256{});

		ASSERT_NO_THROW(tree.check_tree());
		expect_tree_equals(tree, options.set_collision_flags ? expected_tree.dump_with_collisions(expected.flagged) : expected_tree.dump());

		if (options.set_collision_flags)
		{
			// asking again changes nothing
			const OctreeCollisionResult again = query(tree, options);
			ASSERT_EQ(again, result);
			expect_tree_equals(tree, expected_tree.dump_with_collisions(expected.flagged));
			tree.clear_collision_flags();
		}
		expect_tree_equals(tree, expected_tree.dump());
	}

	std::vector<OctreeCollisionOptions> option_variants()
	{
		std::vector<OctreeCollisionOptions> variants;
		for (uint32_t min_level : { 0u, 1u, 3u, 6u })
		{
			OctreeCollisionOptions o;
			o.min_level = min_level;
			o.compute_voxel_flags = true;
			o.count_unknown = true;
			variants.push_back(o);
		}
		OctreeCollisionOptions shifted;
		shifted.offset_x = 5;
		shifted.offset_y = -3;
		shifted.offset_z = 7;
		shifted.compute_voxel_flags = true;
		shifted.count_unknown = true;
		variants.push_back(shifted);
		OctreeCollisionOptions flags_only; // each of the two optional results alone
		flags_only.compute_voxel_flags = true;
		variants.push_back(flags_only);
		OctreeCollisionOptions unknown_only;
		unknown_only.count_unknown = true;
		variants.push_back(unknown_only);
		OctreeCollisionOptions plain; // flags off: no flags, no unknown count
		plain.set_collision_flags = false;
		variants.push_back(plain);
		OctreeCollisionOptions coarse_shifted = shifted;
		coarse_shifted.min_level = 2;
		coarse_shifted.set_collision_flags = false;
		variants.push_back(coarse_shifted);
		// extreme thresholds of the probabilistic tree: -128 would make unknown nodes occupied if the unknown test were missing
		for (int32_t threshold : { -128, -1, 127 })
		{
			OctreeCollisionOptions o;
			o.tree_occupancy_threshold = threshold;
			o.count_unknown = true;
			o.compute_voxel_flags = true;
			variants.push_back(o);
		}
		return variants;
	}
}

TEST_F(VulkanTest, OctreeCollidesWithBitMaps)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(11);
	const Dim3ui dims{ 24, 20, 28 };
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto tree_voxels = random_voxels(rng, 1500, 36, 2, 1, 3); // overlaps the map and sticks out of it
		Octree tree(ctx(), kind);
		tree.build(to_voxels(tree_voxels));
		const ref::OctreeReference expected_tree(to_ref(kind), tree_voxels);

		BitVoxelMap map(ctx(), dims, 1.0f);
		map.insert_coordinate_list(random_coords(rng, 300, dims), 1);
		map.insert_coordinate_list(random_coords(rng, 200, dims), 5);
		map.insert_coordinate_list(random_coords(rng, 100, dims), 40);
		map.insert_coordinate_list(random_coords(rng, 100, dims), 130);
		map.insert_coordinate_list(random_coords(rng, 50, dims), 0); // eBVM_FREE: bit 0 counts as occupied for the tree
		const auto words = map.download_words();

		for (const auto& options : option_variants())
			check_query(tree, expected_tree, probes_of_bit_words(words, dims, options), true, options,
				[&](Octree& t, const OctreeCollisionOptions& o) { return t.collide_with(map, o); }, "bit map");
	}
}

TEST_F(VulkanTest, OctreeCollidesWithProbabilisticMaps)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(12);
	const Dim3ui dims{ 22, 26, 18 };
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto tree_voxels = random_voxels(rng, 1200, 30, 0, 2, 1);
		Octree tree(ctx(), kind);
		tree.build(to_voxels(tree_voxels));
		const ref::OctreeReference expected_tree(to_ref(kind), tree_voxels);

		ProbVoxelMap map(ctx(), dims, 1.0f);
		UpdateOptions update;
		map.insert_coordinate_list(random_coords(rng, 500, dims), eBVM_OCCUPIED, update);
		map.insert_coordinate_list(random_coords(rng, 300, dims), eBVM_OCCUPIED, update); // some voxels twice: larger values
		map.insert_coordinate_list(random_coords(rng, 300, dims), eBVM_FREE, update);
		const auto values = map.download_occupancy();

		for (int32_t threshold : { 50, 1, -20, 127 })
			for (auto options : option_variants())
			{
				options.map_occupancy_threshold = threshold;
				options.compute_voxel_flags = false;
				check_query(tree, expected_tree, probes_of_prob(values, dims, options), false, options,
					[&](Octree& t, const OctreeCollisionOptions& o) { return t.collide_with(map, o); }, "prob map threshold " + std::to_string(threshold));
			}
		OctreeCollisionOptions with_flags;
		with_flags.compute_voxel_flags = true;
		EXPECT_THROW((void)tree.collide_with(map, with_flags), VulkanError);
	}
}

TEST_F(VulkanTest, OctreeCollidesWithBitLists)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(13);
	const Dim3ui dims{ 30, 30, 30 };
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto tree_voxels = random_voxels(rng, 1000, 34, 1, 1, 1);
		Octree tree(ctx(), kind);
		tree.build(to_voxels(tree_voxels));
		const ref::OctreeReference expected_tree(to_ref(kind), tree_voxels);

		BitVoxelList list(ctx(), dims, 1.0f);
		list.insert_coordinate_list(random_coords(rng, 400, dims), 1);
		list.insert_coordinate_list(random_coords(rng, 300, dims), 77);
		list.insert_coordinate_list(random_coords(rng, 200, dims), 200);
		const auto coordinates = list.download_coordinates();
		const auto vectors = list.download_bit_vectors();
		ASSERT_EQ(coordinates.size(), vectors.size());

		for (const auto& options : option_variants())
		{
			Probes probes;
			for (size_t i = 0; i < coordinates.size(); ++i)
			{
				uint32_t any = 0;
				for (uint32_t w : vectors[i])
					any |= w;
				if (any)
					add_probe(probes, int64_t(coordinates[i].x) + options.offset_x, int64_t(coordinates[i].y) + options.offset_y, int64_t(coordinates[i].z) + options.offset_z, vectors[i]);
			}
			check_query(tree, expected_tree, probes, true, options, [&](Octree& t, const OctreeCollisionOptions& o) { return t.collide_with(list, o); }, "bit list");
		}
	}
}

TEST_F(VulkanTest, OctreeCollidesWithMortonIds)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(14);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto tree_voxels = random_voxels(rng, 1500, 40, 100, 100, 100);
		Octree tree(ctx(), kind);
		tree.build(to_voxels(tree_voxels));
		const ref::OctreeReference expected_tree(to_ref(kind), tree_voxels);

		// probes: half inside the occupied cluster, some of them twice, plus ids outside the tree and flags with some zero vectors
		auto probe_voxels = random_voxels(rng, 1500, 44, 98, 98, 98);
		std::vector<uint64_t> ids;
		for (const auto& v : probe_voxels)
			ids.push_back(ref::morton(v.x, v.y, v.z));
		ids.insert(ids.end(), ids.begin(), ids.begin() + 100); // duplicates count twice
		ids.push_back(uint64_t(1) << 42);
		ids.push_back(~uint64_t(0));
		std::vector<BitVector256> flags(ids.size());
		Probes with_flags;
		for (size_t i = 0; i < ids.size(); ++i)
		{
			if (rng() % 5 != 0)
				flags[i][rng() % 8] = uint32_t(rng()) | 1u;
			with_flags.ids.push_back(ids[i]);
			with_flags.flags.push_back(flags[i]);
		}
		Probes occupied_only;
		for (size_t i = 0; i < ids.size(); ++i)
		{
			uint32_t any = 0;
			for (uint32_t w : flags[i])
				any |= w;
			if (any)
			{
				occupied_only.ids.push_back(ids[i]);
				occupied_only.flags.push_back(flags[i]);
			}
		}
		Probes all_probes;
		all_probes.ids = ids;

		for (const auto& base : option_variants())
		{
			OctreeCollisionOptions options = base;
			options.offset_x = options.offset_y = options.offset_z = 0; // ids are not shifted
			check_query(tree, expected_tree, occupied_only, true, options,
				[&](Octree& t, const OctreeCollisionOptions& o) { return t.collide_with_ids(ids, flags, o); }, "ids with flags");
			check_query(tree, expected_tree, all_probes, false, options,
				[&](Octree& t, const OctreeCollisionOptions& o) { return t.collide_with_ids(ids, {}, o); }, "ids without flags");
		}
		EXPECT_THROW((void)tree.collide_with_ids(ids, std::span<const BitVector256>(flags).first(5)), VulkanError);
		EXPECT_EQ(tree.collide_with_ids({}, {}), OctreeCollisionResult{});
	}
}

TEST_F(VulkanTest, OctreeCollisionsDependOnTheNodeWhereTheLookupStops)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	// one occupied voxel at the origin: probes next to it hit it only when the lookup stops at a coarser, partly occupied node
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		tree.build(std::vector<OctreeVoxel>{ { 0, 0, 0 } });
		const std::vector<uint64_t> ids = { ref::morton(0, 0, 0), ref::morton(1, 0, 0), ref::morton(1, 1, 1), ref::morton(2, 0, 0), ref::morton(5, 5, 5), ref::morton(100, 0, 0) };
		for (uint32_t min_level : { 0u, 1u, 2u, 3u, 14u })
		{
			OctreeCollisionOptions o;
			o.min_level = min_level;
			o.set_collision_flags = false;
			const auto result = tree.collide_with_ids(ids, {}, o);
			// a probe collides when it lies inside the box (side 2^min_level) of the occupied voxel; level 14 is the root: everything collides
			uint32_t expected = 0;
			for (uint64_t id : ids)
			{
				const auto v = ref::unmorton(id);
				const uint32_t side = min_level >= 14 ? 16384u : (1u << min_level);
				expected += (v.x < side && v.y < side && v.z < side) || min_level >= 14;
			}
			EXPECT_EQ(result.collisions, expected) << "min_level " << min_level;
		}
	}
}

TEST_F(VulkanTest, OctreeCollisionIgnoresProbesOutsideTheGrid)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	tree.build(std::vector<OctreeVoxel>{ { 0, 0, 0 }, { 16383, 16383, 16383 }, { 0, 16383, 0 }, { 0, 16383, 16383 } });
	BitVoxelMap map(ctx(), Dim3ui{ 4, 4, 4 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 0, 0, 0 }, { 3, 3, 3 } }, 1);
	OctreeCollisionOptions o;
	o.set_collision_flags = false;
	EXPECT_EQ(tree.collide_with(map, o).collisions, 1u); // (0,0,0) hits, (3,3,3) is empty space
	o.offset_x = -1; // (0,0,0) moves to x = -1: outside, not wrapped around to 2^32 - 1
	o.offset_y = 16383;
	EXPECT_EQ(tree.collide_with(map, o).collisions, 0u);
	o.offset_x = 0;
	o.offset_y = 16383;
	o.offset_z = 0;
	EXPECT_EQ(tree.collide_with(map, o).collisions, 1u); // (0,0,0) -> (0,16383,0) is occupied, (3,3,3) -> y 16386 is outside
	o.offset_x = 16383;
	o.offset_y = 16383;
	o.offset_z = 16383;
	EXPECT_EQ(tree.collide_with(map, o).collisions, 1u); // (0,0,0) -> the far corner; the other probe lies outside
	o.offset_x = 16384;
	EXPECT_EQ(tree.collide_with(map, o).collisions, 0u); // x = 16384 would alias to x = 0 of the occupied voxel (0,16383,16383) if it were not rejected
}

TEST_F(VulkanTest, OctreeCollisionFlagsCanBeExtractedAndCleared)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(15);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto tree_voxels = random_voxels(rng, 800, 24, 10, 10, 10);
		Octree tree(ctx(), kind);
		tree.build(to_voxels(tree_voxels));
		const ref::OctreeReference expected_tree(to_ref(kind), tree_voxels);
		std::vector<uint64_t> ids;
		for (const auto& v : random_voxels(rng, 600, 30, 8, 8, 8))
			ids.push_back(ref::morton(v.x, v.y, v.z));
		const auto expected = expected_tree.collide(ids, nullptr, 0);
		ASSERT_GT(expected.collisions, 0u);

		(void)tree.collide_with_ids(ids);
		OctreeExtractOptions only_collisions;
		only_collisions.status_selection.assign(256, 0);
		for (int s = 0; s < 256; ++s)
			only_collisions.status_selection[s] = (s & fmt::kCollision) != 0;
		auto cubes = tree.extract(only_collisions);
		std::set<std::pair<uint32_t, uint64_t>> found;
		for (const auto& c : cubes)
		{
			EXPECT_TRUE(c.status & fmt::kCollision);
			found.insert({ c.level, c.id });
		}
		EXPECT_EQ(found, expected.flagged); // at min_level 0 every flagged node is a leaf or a uniform node, so extract sees all of them
		EXPECT_EQ(cubes.size(), expected.flagged.size());

		tree.clear_collision_flags();
		EXPECT_TRUE(tree.extract(only_collisions).empty());
		expect_tree_equals(tree, expected_tree.dump());
	}
}

TEST_F(VulkanTest, OctreeCollisionOnTheEmptyTree)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		Octree tree(ctx(), kind);
		OctreeCollisionOptions o;
		o.count_unknown = true;
		const std::vector<uint64_t> ids = { 0, 5, 12345 };
		const auto result = tree.collide_with_ids(ids, {}, o);
		EXPECT_EQ(result.collisions, 0u);
		EXPECT_EQ(result.collisions_with_unknown, 3u); // the root is unknown
		EXPECT_NO_THROW(tree.check_tree());
		tree.clear_collision_flags();
	}
}

TEST_F(VulkanTest, OctreeCollisionOfLargeSets)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(16);
	const auto tree_voxels = random_voxels(rng, 1'000'000, 200, 5000, 6000, 7000);
	std::set<uint64_t> occupied;
	for (const auto& v : tree_voxels)
		occupied.insert(ref::morton(v.x, v.y, v.z));
	const Dim3ui dims{ 128, 128, 128 };
	BitVoxelMap map(ctx(), dims, 1.0f);
	map.insert_coordinate_list(random_coords(rng, 300'000, dims), 1);
	const auto words = map.download_words();

	for (const auto kind : kKinds)
	{
		Octree tree(ctx(), kind);
		tree.build(to_voxels(tree_voxels));
		OctreeCollisionOptions o;
		o.offset_x = 5050;
		o.offset_y = 6050;
		o.offset_z = 7050;
		o.set_collision_flags = true;
		o.compute_voxel_flags = true;
		const auto t0 = std::chrono::steady_clock::now();
		const auto result = tree.collide_with(map, o);
		const auto t1 = std::chrono::steady_clock::now();
		uint32_t expected = 0;
		for (uint64_t i = 0; i < dims.voxel_count(); ++i)
		{
			bool any = false;
			for (size_t w = 0; w < 8; ++w)
				any = any || words[8 * i + w];
			if (any && occupied.count(ref::morton(uint32_t(i % dims.x) + 5050, uint32_t((i / dims.x) % dims.y) + 6050, uint32_t(i / (uint64_t(dims.x) * dims.y)) + 7050)))
				++expected;
		}
		std::cout << "[vulkan] octree (" << name(kind) << ") vs map of " << dims.voxel_count() << " voxels: " << result.collisions << " collisions in "
			<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms (" << ctx().info().name << ")" << std::endl;
		EXPECT_EQ(result.collisions, expected);
		EXPECT_GT(result.collisions, 0u);
		EXPECT_NO_THROW(tree.check_tree());
		tree.clear_collision_flags();
		EXPECT_NO_THROW(tree.check_tree());
	}
}
