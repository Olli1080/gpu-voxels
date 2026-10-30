// Slice S6b: Octree build, find, extract and check_tree against the independent CPU reference (tests/reference/octree_reference.h).
#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Morton.h>
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

	std::vector<OctreeVoxel> to_voxels(const std::vector<ref::Voxel>& v)
	{
		std::vector<OctreeVoxel> out;
		for (const auto& x : v)
			out.push_back({ x.x, x.y, x.z });
		return out;
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

	std::vector<ref::Voxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent, uint32_t ox = 0, uint32_t oy = 0, uint32_t oz = 0)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<ref::Voxel> v;
		for (size_t i = 0; i < count; ++i)
			v.push_back({ ox + d(rng), oy + d(rng), oz + d(rng) });
		return v;
	}

	bool same_node(const OctreeNode& a, const ref::Node& b)
	{
		return a.level == b.level && a.id == b.id && a.status == b.status && a.occupancy == b.occupancy && a.has_children == b.has_children;
	}

	bool cube_less(const OctreeCube& a, const OctreeCube& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; }

	/// Builds `voxels` and compares everything the S6b API offers with the reference.
	/// `thorough`: every minimum level x status selection for extract (each combination costs about two traversals with one submit per level),
	/// otherwise three representative combinations.
	void verify_build(Context& ctx, OctreeKind kind, const std::vector<ref::Voxel>& voxels, const std::string& label, std::mt19937_64& rng, bool thorough = false)
	{
		SCOPED_TRACE(std::string(name(kind)) + ": " + label);
		Octree tree(ctx, kind);
		tree.build(to_voxels(voxels));
		const ref::OctreeReference expected(to_ref(kind), voxels);

		ASSERT_NO_THROW(tree.check_tree());
		EXPECT_EQ(tree.voxel_count(), expected.voxel_count());
		EXPECT_EQ(tree.dropped_count(), expected.dropped());

		// memory tree node by node
		const auto dump = tree.dump();
		const auto expected_dump = expected.dump();
		ASSERT_EQ(dump.size(), expected_dump.size());
		for (size_t i = 0; i < dump.size(); ++i)
			ASSERT_TRUE(same_node(dump[i], expected_dump[i])) << "node " << i << ": level " << dump[i].level << " id " << dump[i].id
				<< " status " << int(dump[i].status) << " occupancy " << int(dump[i].occupancy) << " children " << dump[i].has_children
				<< " / expected level " << expected_dump[i].level << " id " << expected_dump[i].id << " status " << int(expected_dump[i].status)
				<< " occupancy " << int(expected_dump[i].occupancy) << " children " << expected_dump[i].has_children;

		// memory usage: blocks of 8 per level (the root level block holds the root and seven unused nodes)
		const auto stats = tree.memory_usage();
		for (uint32_t level = 0; level < Octree::kLevels; ++level)
			EXPECT_EQ(stats.blocks[level], level == 14 ? 1u : expected.node_count(level) / 8) << "level " << level;
		uint64_t bytes = 0;
		for (uint32_t level = 0; level < Octree::kLevels; ++level)
			bytes += stats.blocks[level] * fmt::block_bytes(kind, level);
		EXPECT_EQ(stats.bytes, bytes);

		// find: every voxel, neighbours, random ids, ids outside the tree
		std::vector<uint64_t> ids;
		for (uint64_t id : expected.voxel_ids())
		{
			ids.push_back(id);
			ids.push_back(id ^ 1);
			ids.push_back(id ^ 8);
		}
		std::uniform_int_distribution<uint64_t> any(0, (uint64_t(1) << 42) - 1);
		for (int i = 0; i < 300; ++i)
			ids.push_back(any(rng));
		ids.push_back(0);
		ids.push_back((uint64_t(1) << 42) - 1);
		ids.push_back(uint64_t(1) << 42);
		ids.push_back(~uint64_t(0));
		const auto found = tree.find(ids);
		ASSERT_EQ(found.size(), ids.size());
		for (size_t i = 0; i < ids.size(); ++i)
		{
			if (ids[i] >= (uint64_t(1) << 42))
			{
				ASSERT_EQ(found[i].level, OctreeLookup::kOutside) << ids[i];
				continue;
			}
			const ref::Node& e = expected.lookup(ids[i]);
			ASSERT_TRUE(found[i].level == e.level && found[i].status == e.status && found[i].occupancy == e.occupancy)
				<< "id " << ids[i] << ": level " << found[i].level << " status " << int(found[i].status) << " occupancy " << int(found[i].occupancy)
				<< " / expected level " << e.level << " status " << int(e.status) << " occupancy " << int(e.occupancy);
		}

		// extract: different minimum levels and status selections
		std::vector<uint8_t> occupied_only(256, 0), unknown_only(256, 0);
		for (int s = 0; s < 256; ++s)
		{
			occupied_only[s] = (s & fmt::kOccupied) != 0;
			unknown_only[s] = (s & fmt::kOccupancyMask) == fmt::kUnknown;
		}
		const std::vector<uint8_t> selections[] = { {}, occupied_only, unknown_only };
		std::vector<std::pair<uint32_t, size_t>> combinations = { { 0, 0 }, { 1, 1 }, { 5, 2 } };
		if (thorough)
		{
			combinations.clear();
			for (uint32_t min_level : { 0u, 1u, 2u, 5u, 13u })
				for (size_t sel = 0; sel < 3; ++sel)
					combinations.emplace_back(min_level, sel);
		}
		for (const auto& [min_level, sel] : combinations)
			{
				OctreeExtractOptions options;
				options.min_level = min_level;
				options.status_selection = selections[sel];
				auto cubes = tree.extract(options);
				std::sort(cubes.begin(), cubes.end(), cube_less);
				auto expected_nodes = expected.extract(selections[sel], min_level);
				std::sort(expected_nodes.begin(), expected_nodes.end(), [](const ref::Node& a, const ref::Node& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
				ASSERT_EQ(cubes.size(), expected_nodes.size()) << "min_level " << min_level << " selection " << sel;
				for (size_t i = 0; i < cubes.size(); ++i)
					ASSERT_TRUE(cubes[i].level == expected_nodes[i].level && cubes[i].id == expected_nodes[i].id && cubes[i].status == expected_nodes[i].status &&
						cubes[i].occupancy == expected_nodes[i].occupancy) << "min_level " << min_level << " selection " << sel << " cube " << i;
			}
	}
}

TEST_F(VulkanTest, OctreeRequiresDeviceAddresses)
{
	ContextCreateInfo info;
	info.require_validation = std::getenv("GVL_ALLOW_NO_VALIDATION") == nullptr;
	info.enable_buffer_device_address = false;
	Context context(info);
	EXPECT_THROW(Octree(context, OctreeKind::Deterministic), VulkanError);
}

TEST_F(VulkanTest, OctreeStartsEmptyAndClearsBackToEmpty)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(1);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		EXPECT_EQ(tree.voxel_count(), 0u);
		EXPECT_NO_THROW(tree.check_tree());
		EXPECT_EQ(tree.memory_usage().bytes, fmt::inner_block_bytes(kind));
		EXPECT_EQ(tree.memory_usage().nodes(), 8u);
		EXPECT_TRUE(tree.extract().empty());

		tree.build(std::vector<OctreeVoxel>{});
		verify_build(ctx(), kind, {}, "empty input", rng);

		tree.build(to_voxels(cube(0, 0, 0, 3)));
		EXPECT_EQ(tree.voxel_count(), 27u);
		tree.clear();
		EXPECT_EQ(tree.voxel_count(), 0u);
		EXPECT_EQ(tree.dropped_count(), 0u);
		EXPECT_EQ(tree.memory_usage().bytes, fmt::inner_block_bytes(kind));
		EXPECT_NO_THROW(tree.check_tree());
		const auto dump = tree.dump();
		ASSERT_EQ(dump.size(), 1u);
		EXPECT_EQ(dump[0].status, fmt::kUnknown);
		EXPECT_EQ(dump[0].occupancy, kind == OctreeKind::Probabilistic ? -128 : 0);
		const auto found = tree.find(std::vector<uint64_t>{ 0, 12345 });
		ASSERT_EQ(found.size(), 2u);
		EXPECT_EQ(found[0].level, 14u);
		EXPECT_EQ(found[1].level, 14u);
	}
}

TEST_F(VulkanTest, OctreeBuildMatchesHandDerivedTrees)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(2);
	for (const auto kind : kKinds)
	{
		verify_build(ctx(), kind, { { 0, 0, 0 } }, "single voxel", rng, true);
		verify_build(ctx(), kind, { { 16383, 16383, 16383 } }, "far corner", rng);
		verify_build(ctx(), kind, { { 5, 5, 5 }, { 5, 5, 5 }, { 5, 5, 5 } }, "duplicates", rng);
		verify_build(ctx(), kind, cube(0, 0, 0, 2), "full 2x2x2", rng, true);
		verify_build(ctx(), kind, cube(4, 4, 4, 4), "full aligned 4x4x4", rng, true);
		verify_build(ctx(), kind, cube(3, 5, 6, 4), "full unaligned 4x4x4", rng);
		verify_build(ctx(), kind, { { 0, 0, 0 }, { 8192, 0, 0 } }, "two voxels meeting at the root", rng, true);
		verify_build(ctx(), kind, { { 0, 0, 0 }, { 16383, 16383, 16383 }, { 16383, 0, 0 }, { 0, 16383, 0 }, { 0, 0, 16383 } }, "corners", rng);
		verify_build(ctx(), kind, { { 1023, 1024, 0 }, { 1024, 1023, 0 }, { 1023, 1023, 1023 }, { 1024, 1024, 1024 } }, "around 1024", rng);
		verify_build(ctx(), kind, { { 16384, 0, 0 }, { 0, 16384, 0 }, { 0, 0, 16384 }, { 7, 7, 7 }, { 1u << 20, 1, 1 }, { 0xFFFFFFFFu, 0, 0 } }, "dropped voxels", rng);
		verify_build(ctx(), kind, { { 16384, 0, 0 }, { 0, 16384, 0 } }, "only dropped voxels", rng);
	}
}

TEST_F(VulkanTest, OctreeBuildMatchesTheReferenceOnRandomSets)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(3);
	for (const auto kind : kKinds)
	{
		for (size_t count : { size_t(2), size_t(10), size_t(200), size_t(2500) })
		{
			verify_build(ctx(), kind, random_voxels(rng, count, 16384), "sparse " + std::to_string(count), rng);
			verify_build(ctx(), kind, random_voxels(rng, count, 12, 100, 200, 300), "dense cluster " + std::to_string(count), rng);
			verify_build(ctx(), kind, random_voxels(rng, count, 8, 16376, 16376, 16376), "far corner cluster " + std::to_string(count), rng);
		}
		auto holes = cube(32, 32, 32, 8);
		for (int i = 0; i < 40; ++i)
			holes.erase(holes.begin() + long(rng() % holes.size()));
		verify_build(ctx(), kind, holes, "cube with holes", rng, true);
		verify_build(ctx(), kind, cube(60, 70, 80, 16), "full 16 cube", rng);
	}
}

TEST_F(VulkanTest, OctreeBuildFromIdsMatchesBuildFromCoordinates)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(4);
	for (const auto kind : kKinds)
	{
		const auto voxels = random_voxels(rng, 500, 40, 1000, 2000, 3000);
		std::vector<uint64_t> ids;
		for (const auto& v : voxels)
			ids.push_back(gpu_voxels::vulkan::morton::code60(v.x, v.y, v.z));
		ids.push_back(uint64_t(1) << 42);
		ids.push_back((uint64_t(1) << 42) + 5);
		ids.push_back(~uint64_t(0));

		Octree from_ids(ctx(), kind), from_coordinates(ctx(), kind);
		from_ids.build_from_ids(ids);
		from_coordinates.build(to_voxels(voxels));
		EXPECT_EQ(from_ids.dropped_count(), 3u);
		EXPECT_EQ(from_ids.voxel_count(), from_coordinates.voxel_count());
		EXPECT_EQ(from_ids.dump(), from_coordinates.dump());
		EXPECT_NO_THROW(from_ids.check_tree());

		// ids over the whole 42 bit range (all bits take part in the sort)
		auto wide = random_voxels(rng, 800, 16384);
		wide.push_back({ 16383, 16383, 16383 }); // id 2^42 - 1
		wide.push_back({ 0, 0, 8192 });          // id 2^41
		std::vector<uint64_t> wide_ids;
		for (const auto& v : wide)
			wide_ids.push_back(ref::morton(v.x, v.y, v.z));
		from_ids.build_from_ids(wide_ids);
		from_coordinates.build(to_voxels(wide));
		EXPECT_EQ(from_ids.dump(), from_coordinates.dump());
		EXPECT_NO_THROW(from_ids.check_tree());

		from_ids.build_from_ids(std::vector<uint64_t>{ uint64_t(1) << 42 });
		EXPECT_EQ(from_ids.voxel_count(), 0u);
		EXPECT_EQ(from_ids.dropped_count(), 1u);
		EXPECT_EQ(from_ids.dump().size(), 1u);
	}
}

TEST_F(VulkanTest, OctreeRebuildReplacesTheOldTree)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(5);
	Octree tree(ctx(), OctreeKind::Deterministic);
	tree.build(to_voxels(random_voxels(rng, 3000, 16384)));
	const auto big = tree.memory_usage().bytes;
	const auto small_set = random_voxels(rng, 5, 16384);
	tree.build(to_voxels(small_set));
	const ref::OctreeReference expected(ref::Kind::Deterministic, small_set);
	EXPECT_LT(tree.memory_usage().bytes, big);
	EXPECT_EQ(tree.dump().size(), expected.node_count());
	EXPECT_EQ(tree.dropped_count(), 0u);
	EXPECT_NO_THROW(tree.check_tree());
}

TEST_F(VulkanTest, OctreeTwoDimensionalDispatch)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(6);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		const auto voxels = random_voxels(rng, 3000, 30, 10, 20, 30);
		Octree tree(ctx(), kind);
		tree.set_max_groups_x(3); // 3000 / 64 = 47 groups -> a 3 x 16 grid
		tree.build(to_voxels(voxels));
		const ref::OctreeReference expected(to_ref(kind), voxels);
		ASSERT_NO_THROW(tree.check_tree());
		const auto dump = tree.dump();
		ASSERT_EQ(dump.size(), expected.node_count());
		const auto expected_dump = expected.dump();
		for (size_t i = 0; i < dump.size(); ++i)
			ASSERT_TRUE(same_node(dump[i], expected_dump[i])) << i;
		std::vector<uint64_t> ids(expected.voxel_ids().begin(), expected.voxel_ids().end());
		const auto found = tree.find(ids);
		for (size_t i = 0; i < ids.size(); ++i)
			ASSERT_TRUE(found[i].level == expected.lookup(ids[i]).level && found[i].status == expected.lookup(ids[i]).status) << i;
		EXPECT_EQ(tree.extract().size(), expected.extract({}, 0).size());
	}
}

TEST_F(VulkanTest, OctreeMemoryUsageOfOneVoxel)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		Octree tree(ctx(), kind);
		tree.build(std::vector<OctreeVoxel>{ { 3, 4, 5 } });
		const auto stats = tree.memory_usage();
		for (uint32_t level = 0; level < Octree::kLevels; ++level)
			EXPECT_EQ(stats.blocks[level], 1u) << level;
		EXPECT_EQ(stats.nodes(), 15u * 8u);
		EXPECT_EQ(stats.bytes, kind == OctreeKind::Probabilistic ? 16u + 14u * 96u : 8u + 14u * 64u);
		EXPECT_EQ(tree.chunk(0).size(), fmt::leaf_block_bytes(kind));
		EXPECT_EQ(tree.chunk(14).size(), fmt::inner_block_bytes(kind));
		EXPECT_EQ(tree.chunk(7).device_address() % 16, 0u);
	}
}

TEST_F(VulkanTest, OctreeExtractAndFindRejectInvalidArguments)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	tree.build(std::vector<OctreeVoxel>{ { 1, 2, 3 } });
	OctreeExtractOptions bad_level;
	bad_level.min_level = 14;
	EXPECT_THROW((void)tree.extract(bad_level), VulkanError);
	OctreeExtractOptions bad_selection;
	bad_selection.status_selection.assign(255, 1);
	EXPECT_THROW((void)tree.extract(bad_selection), VulkanError);
	EXPECT_TRUE(tree.find(std::vector<uint64_t>{}).empty());
	EXPECT_THROW((void)tree.chunk(15), VulkanError);
	Octree empty(ctx(), OctreeKind::Deterministic);
	EXPECT_THROW((void)empty.chunk(3), VulkanError);
}

TEST_F(VulkanTest, OctreeBuildOfOneMillionVoxels)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(7);
	const auto voxels = random_voxels(rng, 1'000'000, 200, 5000, 6000, 7000);
	std::set<uint64_t> distinct;
	for (const auto& v : voxels)
		distinct.insert(ref::morton(v.x, v.y, v.z));
	for (const auto kind : kKinds)
	{
		Octree tree(ctx(), kind);
		const auto t0 = std::chrono::steady_clock::now();
		tree.build(to_voxels(voxels));
		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "[vulkan] octree build (" << name(kind) << ") of 1 M voxels (" << distinct.size() << " distinct) incl. transfers: "
			<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms, " << tree.memory_usage().bytes / (1024 * 1024) << " MiB ("
			<< ctx().info().name << ")" << std::endl;
		EXPECT_EQ(tree.voxel_count(), distinct.size());
		EXPECT_NO_THROW(tree.check_tree());
		std::vector<uint64_t> probe;
		size_t k = 0;
		for (uint64_t id : distinct)
			if (k++ % 997 == 0)
				probe.push_back(id);
		const auto found = tree.find(probe);
		for (size_t i = 0; i < probe.size(); ++i)
		{
			ASSERT_EQ(found[i].level, 0u) << i;
			if (kind == OctreeKind::Deterministic)
				ASSERT_EQ(found[i].status, fmt::kOccupied | fmt::kStatic);
			else
				ASSERT_EQ(found[i].occupancy, 127);
		}
		OctreeExtractOptions leaves;
		leaves.status_selection.assign(256, 0);
		for (int s = 0; s < 256; ++s)
			leaves.status_selection[s] = kind == OctreeKind::Deterministic ? (s & fmt::kOccupied) != 0 : s == fmt::kStatic;
		auto cubes = tree.extract(leaves);
		size_t voxel_cubes = 0;
		for (const auto& c : cubes)
			voxel_cubes += c.level == 0;
		EXPECT_EQ(voxel_cubes, distinct.size());
	}
}

TEST_F(VulkanTest, OctreeNodeMemoryIsExactlyTheHostFormat)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(8);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		tree.build(to_voxels(random_voxels(rng, 400, 16384)));
		auto more = random_voxels(rng, 400, 20, 77, 88, 99);
		tree.build(to_voxels(more));
		for (uint32_t level = 0; level < Octree::kLevels; ++level)
		{
			std::vector<uint8_t> bytes(tree.chunk(level).size());
			tree.chunk(level).download(bytes.data(), bytes.size());
			const size_t block_size = fmt::block_bytes(kind, level);
			ASSERT_EQ(bytes.size() % block_size, 0u);
			for (size_t offset = 0; offset < bytes.size(); offset += block_size)
			{
				std::vector<uint8_t> rebuilt(block_size, 0);
				for (uint32_t slot = 0; slot < 8; ++slot)
				{
					if (level == 0)
						fmt::encode_leaf(kind, rebuilt.data(), slot, fmt::decode_leaf(kind, bytes.data() + offset, slot));
					else
					{
						const auto node = fmt::decode_inner(kind, bytes.data() + offset, slot);
						ASSERT_EQ(node.flags, 0u) << "a freshly built tree has no flags set";
						fmt::encode_inner(kind, rebuilt.data(), slot, node);
					}
				}
				ASSERT_TRUE(std::equal(rebuilt.begin(), rebuilt.end(), bytes.begin() + long(offset))) << "level " << level << " block at byte " << offset
					<< " holds bits the host format does not know";
			}
		}
	}
}

TEST_F(VulkanTest, OctreeFlagsDoNotDisturbStatusOrPointers)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(9);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind);
		tree.build(to_voxels(random_voxels(rng, 600, 64, 300, 400, 500)));
		const auto dump_before = tree.dump();
		std::vector<uint64_t> ids;
		for (const auto& n : dump_before)
			if (n.level == 0)
				ids.push_back(n.id);
		const auto found_before = tree.find(ids);
		auto cubes_before = tree.extract();

		// set every flag bit of every inner node: the update flags of the dynamic operations (slice S6d) share the word with status and pointer
		for (uint32_t level = 1; level < Octree::kLevels; ++level)
		{
			std::vector<uint8_t> bytes(tree.chunk(level).size());
			tree.chunk(level).download(bytes.data(), bytes.size());
			for (size_t offset = 0; offset < bytes.size(); offset += fmt::inner_block_bytes(kind))
				for (uint32_t slot = 0; slot < 8; ++slot)
				{
					auto node = fmt::decode_inner(kind, bytes.data() + offset, slot);
					node.flags = 0xFF;
					fmt::encode_inner(kind, bytes.data() + offset, slot, node);
				}
			tree.chunk(level).upload(bytes.data(), bytes.size());
		}
		auto dump_after = tree.dump();
		ASSERT_EQ(dump_after.size(), dump_before.size());
		EXPECT_EQ(dump_after, dump_before);
		EXPECT_NO_THROW(tree.check_tree());
		EXPECT_EQ(tree.find(ids), found_before);
		auto cubes_after = tree.extract();
		std::sort(cubes_before.begin(), cubes_before.end(), cube_less);
		std::sort(cubes_after.begin(), cubes_after.end(), cube_less);
		EXPECT_EQ(cubes_after, cubes_before);
	}
}

// ---------------------------------------------------------------------------------------------------- check_tree

namespace
{
	struct Located { uint64_t block; uint32_t slot; fmt::HostNode node; };

	/// The `nth` inner node of `level` that has a child block.
	Located nth_with_children(Octree& tree, OctreeKind kind, uint32_t level, size_t nth)
	{
		std::vector<uint8_t> bytes(tree.chunk(level).size());
		tree.chunk(level).download(bytes.data(), bytes.size());
		const size_t block_size = fmt::inner_block_bytes(kind);
		size_t seen = 0;
		for (uint64_t b = 0; b < bytes.size() / block_size; ++b)
			for (uint32_t s = 0; s < 8; ++s)
			{
				const auto node = fmt::decode_inner(kind, bytes.data() + b * block_size, s);
				if (node.child != 0 && seen++ == nth)
					return { b, s, node };
			}
		ADD_FAILURE() << "no such node";
		return {};
	}

	void modify_inner(Octree& tree, OctreeKind kind, uint32_t level, const Located& at, const fmt::HostNode& node)
	{
		std::vector<uint8_t> block(fmt::inner_block_bytes(kind));
		const uint64_t offset = at.block * block.size();
		tree.chunk(level).download(block.data(), block.size(), offset);
		fmt::encode_inner(kind, block.data(), at.slot, node);
		tree.chunk(level).upload(block.data(), block.size(), offset);
	}

	void expect_problem(Octree& tree, const std::string& needle)
	{
		try
		{
			tree.check_tree();
			ADD_FAILURE() << "check_tree accepted a corrupt tree (expected: " << needle << ")";
		}
		catch (const VulkanError& e)
		{
			EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << "message was: " << e.what();
		}
	}

	Octree corruptible_tree(Context& ctx, OctreeKind kind)
	{
		Octree tree(ctx, kind);
		tree.build(std::vector<OctreeVoxel>{ { 0, 0, 0 }, { 1, 0, 0 }, { 100, 3, 9 }, { 4000, 4000, 4000 }, { 4001, 4000, 4000 } });
		tree.check_tree();
		return tree;
	}
}

TEST_F(VulkanTest, FindStopsAtANodeWhoseChildPointerIsMissing)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		// PART without a child block is corrupt (check_tree says so), but a lookup must not dereference the null pointer:
		// that would lose the device and with it every later test
		Octree tree = corruptible_tree(ctx(), kind);
		auto at = nth_with_children(tree, kind, 5, 0);
		at.node.child = 0;
		modify_inner(tree, kind, 5, at, at.node);
		const auto found = tree.find(std::vector<uint64_t>{ 0 });
		ASSERT_EQ(found.size(), 1u);
		EXPECT_EQ(found[0].level, 5u);
		EXPECT_EQ(found[0].status, at.node.status);
	}
}

TEST_F(VulkanTest, OctreeCollisionStopsAtANodeWhoseChildPointerIsMissing)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		// same corruption as in FindStopsAtANodeWhoseChildPointerIsMissing: the walk must stop at the node instead of following a null pointer
		Octree tree = corruptible_tree(ctx(), kind);
		auto at = nth_with_children(tree, kind, 5, 0);
		at.node.child = 0;
		modify_inner(tree, kind, 5, at, at.node);
		OctreeCollisionOptions options;
		options.set_collision_flags = false;
		const auto result = tree.collide_with_ids(std::vector<uint64_t>{ 0 }, {}, options);
		EXPECT_EQ(result.collisions, 1u); // the node on level 5 is occupied
	}
}

TEST_F(VulkanTest, CheckTreeFindsCorruptedPointersAndStatus)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		{ // child pointer far outside every chunk
			Octree tree = corruptible_tree(ctx(), kind);
			auto at = nth_with_children(tree, kind, 5, 0);
			at.node.child += uint64_t(1) << 30;
			modify_inner(tree, kind, 5, at, at.node);
			expect_problem(tree, "outside the chunk");
		}
		{ // pointer inside the chunk but not at a block boundary
			Octree tree = corruptible_tree(ctx(), kind);
			auto at = nth_with_children(tree, kind, 5, 0);
			at.node.child += 4;
			modify_inner(tree, kind, 5, at, at.node);
			expect_problem(tree, "not aligned");
		}
		{ // two nodes share one child block (and the other block becomes unreachable)
			Octree tree = corruptible_tree(ctx(), kind);
			const auto first = nth_with_children(tree, kind, 6, 0);
			auto second = nth_with_children(tree, kind, 6, 1);
			second.node.child = first.node.child;
			modify_inner(tree, kind, 6, second, second.node);
			expect_problem(tree, "referenced a second time");
		}
		{ // a node loses its children but keeps PART: also leaves an unreachable block behind
			Octree tree = corruptible_tree(ctx(), kind);
			auto at = nth_with_children(tree, kind, 3, 0);
			at.node.child = 0;
			modify_inner(tree, kind, 3, at, at.node);
			expect_problem(tree, "PART without a child block");
			expect_problem(tree, "not reachable");
		}
		{ // status does not follow from the children
			Octree tree = corruptible_tree(ctx(), kind);
			auto at = nth_with_children(tree, kind, 9, 0);
			at.node.status = at.node.status ^ fmt::kOccupied;
			modify_inner(tree, kind, 9, at, at.node);
			expect_problem(tree, "differ from the rule");
		}
		if (kind == OctreeKind::Probabilistic)
		{ // occupancy does not follow from the children
			Octree tree = corruptible_tree(ctx(), kind);
			auto at = nth_with_children(tree, kind, 9, 0);
			at.node.occupancy = 3;
			modify_inner(tree, kind, 9, at, at.node);
			expect_problem(tree, "differ from the rule");
		}
		{ // LAST_LEVEL on a node that is not on level 1
			Octree tree = corruptible_tree(ctx(), kind);
			auto at = nth_with_children(tree, kind, 5, 0);
			at.node.status |= fmt::kLastLevel;
			modify_inner(tree, kind, 5, at, at.node);
			expect_problem(tree, "LAST_LEVEL");
		}
		{ // level 1 node without LAST_LEVEL
			Octree tree = corruptible_tree(ctx(), kind);
			auto at = nth_with_children(tree, kind, 1, 0);
			at.node.status &= uint8_t(~fmt::kLastLevel);
			modify_inner(tree, kind, 1, at, at.node);
			expect_problem(tree, "LAST_LEVEL");
		}
		{ // PART bit on a leaf
			Octree tree = corruptible_tree(ctx(), kind);
			std::vector<uint8_t> leaves(tree.chunk(0).size());
			tree.chunk(0).download(leaves.data(), leaves.size());
			leaves[0] |= fmt::kPart;
			tree.chunk(0).upload(leaves.data(), leaves.size());
			expect_problem(tree, "leaf has the PART");
		}
		{ // LAST_LEVEL on a leaf
			Octree tree = corruptible_tree(ctx(), kind);
			std::vector<uint8_t> leaves(tree.chunk(0).size());
			tree.chunk(0).download(leaves.data(), leaves.size());
			auto leaf = fmt::decode_leaf(kind, leaves.data(), 1);
			leaf.status |= fmt::kLastLevel;
			fmt::encode_leaf(kind, leaves.data(), 1, leaf);
			tree.chunk(0).upload(leaves.data(), leaves.size());
			expect_problem(tree, "leaf has the PART or LAST_LEVEL");
		}
		{ // a pointer into a level that has no chunk is impossible to build, but a chunk of the wrong level is found by the range check
			Octree tree = corruptible_tree(ctx(), kind);
			auto at = nth_with_children(tree, kind, 5, 0);
			at.node.child = tree.chunk(7).device_address(); // right range for level 7, but the node expects level 4
			modify_inner(tree, kind, 5, at, at.node);
			expect_problem(tree, "outside the chunk of level 4");
		}
	}
}
