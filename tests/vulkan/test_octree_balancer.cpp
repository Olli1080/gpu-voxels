// Slice S6c (third part): the load balancing framework (persistent work groups with stacks, port of the CUDA load balancer) must give
// exactly the results of the level-synchronous wavefront and of the independent reference, also with tiny configurations that force
// many redistributions of the work stacks.
#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
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

	/// Configurations that can hold the work of the test trees: the default, fewer groups and tiny stacks that force many
	/// redistributions (a stack stops at stack_size - 56 items, so the groups together must be able to hold the pending items).
	std::vector<OctreeBalancerConfig> configs()
	{
		std::vector<OctreeBalancerConfig> c;
		c.push_back({});                          // 128 tasks, 700 items
		c.push_back({ 16, 700, 2.0f / 3.0f });
		c.push_back({ 64, 64, 0.5f });            // threshold 8 items per stack: rebalancing all the time
		c.push_back({ 96, 80, 0.9f });
		return c;
	}

	bool same(const std::vector<OctreeNode>& a, const std::vector<OctreeNode>& b) { return a == b; }

	std::vector<OctreeCube> sorted(std::vector<OctreeCube> cubes)
	{
		std::sort(cubes.begin(), cubes.end(), [](const OctreeCube& a, const OctreeCube& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
		return cubes;
	}
}

TEST_F(VulkanTest, BalancedTreeCollisionEqualsTheWavefrontAndTheReference)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(31);
	struct Case { std::string label; std::vector<ref::Voxel> a, b; };
	std::vector<Case> cases;
	cases.push_back({ "dense clusters", random_voxels(rng, 500, 24, 100, 100, 100), random_voxels(rng, 500, 24, 108, 108, 108) });
	cases.push_back({ "sparse", random_voxels(rng, 300, 16384), random_voxels(rng, 300, 16384) });
	cases.push_back({ "cubes (uniform nodes)", cube(0, 0, 0, 8), cube(4, 4, 4, 8) });
	cases.push_back({ "single voxels", { { 0, 0, 0 } }, { { 0, 0, 0 } } });
	cases.push_back({ "empty", {}, cube(0, 0, 0, 4) });
	cases.push_back({ "large overlap", random_voxels(rng, 1500, 40), random_voxels(rng, 1500, 40, 20, 20, 20) });
	// uniform nodes against nodes with leaves, in both directions (the leaves of one side are compared with the uniform node)
	const std::vector<ref::Voxel> few = { { 3, 3, 3 }, { 9, 9, 9 }, { 4, 4, 4 }, { 4, 5, 6 } };
	cases.push_back({ "cube vs single voxels", cube(0, 0, 0, 8), few });
	cases.push_back({ "single voxels vs cube", few, cube(0, 0, 0, 8) });
	auto holes = cube(32, 32, 32, 8);
	for (int i = 0; i < 60; ++i)
		holes.erase(holes.begin() + long(rng() % holes.size()));
	cases.push_back({ "cube with holes vs cluster", holes, random_voxels(rng, 500, 14, 30, 30, 30) });
	cases.push_back({ "cluster vs cube with holes", random_voxels(rng, 500, 14, 30, 30, 30), holes });
	for (const auto kind : kKinds)
		for (const auto& c : cases)
		{
			SCOPED_TRACE(std::string(name(kind)) + ": " + c.label);
			const ref::OctreeReference ref_a(to_ref(kind), c.a), ref_b(to_ref(kind), c.b);
			Octree a(ctx(), kind), b(ctx(), kind);
			a.build(to_voxels(c.a));
			b.build(to_voxels(c.b));
			const bool small_case = c.label != "large overlap";
			for (const auto& config : configs())
			{
				a.set_balancer_config(config);
				const bool default_config = config.tasks == 128;
				for (uint32_t min_level : { 0u, 1u, 3u, 13u })
					for (const bool mark : { true, false })
					{
						if (!default_config && (!small_case || !mark || (min_level != 0 && min_level != 3)))
							continue; // the other configurations are checked on the small cases with marks and two minimum levels
						SCOPED_TRACE("tasks " + std::to_string(config.tasks) + " stack " + std::to_string(config.stack_size) + " min_level " + std::to_string(min_level) + " mark " + std::to_string(mark));
						OctreeTreeCollisionOptions options;
						options.min_level = min_level;
						options.mark_collisions = mark;
						options.use_load_balancer = true;
						const uint64_t volume = a.collide_with(b, options);
						const auto expected = ref_a.collide_tree(ref_b, min_level, 10, mark);
						ASSERT_EQ(volume, expected.voxels);
						const auto dump_a = a.dump(), dump_b = b.dump();
						ASSERT_NO_THROW(a.check_tree());
						ASSERT_NO_THROW(b.check_tree());
						ASSERT_TRUE(same(dump_a, [&] { std::vector<OctreeNode> n; for (const auto& x : ref_a.dump_with_collisions(expected.flagged_this)) n.push_back({ x.level, x.id, x.status, x.occupancy, x.has_children }); return n; }()));
						ASSERT_TRUE(same(dump_b, [&] { std::vector<OctreeNode> n; for (const auto& x : ref_b.dump_with_collisions(expected.flagged_other)) n.push_back({ x.level, x.id, x.status, x.occupancy, x.has_children }); return n; }()));
						a.clear_collision_flags();
						b.clear_collision_flags();

						options.use_load_balancer = false; // the wavefront gives the same volume
						ASSERT_EQ(a.collide_with(b, options), volume);
						a.clear_collision_flags();
						b.clear_collision_flags();
					}
			}
		}
}

TEST_F(VulkanTest, BalancedTreeCollisionWithUniformAndCorruptRoots)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(37);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		// forged roots as in the wavefront test: an occupied root without children is a uniform node for the whole grid; a root with the
		// PART flag but no child block (corrupt) must be treated alike and never followed
		for (const bool with_part_flag : { false, true })
		{
			Octree forged(ctx(), kind), other(ctx(), kind);
			forged.build(std::vector<OctreeVoxel>{ { 1, 2, 3 } });
			std::vector<uint8_t> block(fmt::inner_block_bytes(kind));
			forged.chunk(14).download(block.data(), block.size());
			fmt::HostNode root = fmt::decode_inner(kind, block.data(), 0);
			root.status = uint8_t(fmt::kOccupied | fmt::kStatic | (with_part_flag ? fmt::kPart : 0));
			root.child = 0;
			root.occupancy = kind == OctreeKind::Probabilistic ? fmt::kProbMax : int8_t(0);
			fmt::encode_inner(kind, block.data(), 0, root);
			forged.chunk(14).upload(block.data(), block.size());

			const auto voxels = random_voxels(rng, 400, 30, 600, 700, 800);
			other.build(to_voxels(voxels));
			OctreeTreeCollisionOptions options;
			options.mark_collisions = false;
			options.use_load_balancer = true;
			const uint64_t expected = ref::OctreeReference(to_ref(kind), voxels).voxel_count();
			EXPECT_EQ(forged.collide_with(other, options), expected) << "PART flag " << with_part_flag;
			EXPECT_EQ(other.collide_with(forged, options), expected) << "PART flag " << with_part_flag;
		}
	}
}

TEST_F(VulkanTest, BalancedExtractEqualsTheWavefront)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(32);
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		auto voxels = random_voxels(rng, 1200, 30, 50, 60, 70);
		const auto block = cube(200, 200, 200, 8);
		voxels.insert(voxels.end(), block.begin(), block.end()); // uniform nodes with kept children
		voxels.push_back({ 16383, 16383, 16383 });
		Octree tree(ctx(), kind);
		tree.build(to_voxels(voxels));
		const ref::OctreeReference expected_tree(to_ref(kind), voxels);
		std::vector<uint8_t> occupied(256, 0), unknown(256, 0);
		for (int s = 0; s < 256; ++s)
		{
			occupied[s] = (s & fmt::kOccupied) != 0;
			unknown[s] = (s & fmt::kOccupancyMask) == fmt::kUnknown;
		}
		const std::vector<uint8_t> selections[] = { {}, occupied, unknown };
		for (const auto& config : configs())
		{
			tree.set_balancer_config(config);
			for (uint32_t min_level : { 0u, 1u, 4u, 13u })
				for (const auto& selection : selections)
				{
					if (config.tasks != 128 && (min_level == 1 || min_level == 13))
						continue;
					OctreeExtractOptions options;
					options.min_level = min_level;
					options.status_selection = selection;
					options.use_load_balancer = true;
					const auto balanced = sorted(tree.extract(options));
					options.use_load_balancer = false;
					const auto wavefront = sorted(tree.extract(options));
					ASSERT_EQ(balanced.size(), wavefront.size()) << "tasks " << config.tasks << " min_level " << min_level;
					ASSERT_TRUE(balanced == wavefront) << "tasks " << config.tasks << " stack " << config.stack_size << " min_level " << min_level;
					auto expected = expected_tree.extract(selection, min_level);
					ASSERT_EQ(balanced.size(), expected.size());
				}
		}
	}
}

TEST_F(VulkanTest, BalancerRedistributesTheWorkWithSmallStacks)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(33);
	Octree a(ctx(), OctreeKind::Deterministic), b(ctx(), OctreeKind::Deterministic);
	const auto va = random_voxels(rng, 2000, 40, 10, 10, 10);
	const auto vb = random_voxels(rng, 2000, 40, 20, 20, 20);
	a.build(to_voxels(va));
	b.build(to_voxels(vb));
	OctreeTreeCollisionOptions options;
	options.mark_collisions = false;
	options.use_load_balancer = true;

	a.set_balancer_config({ 64, 64, 0.5f });
	const uint64_t small_stacks = a.collide_with(b, options);
	const auto small_stats = a.balancer_stats();
	a.set_balancer_config({ 1, 700, 1.0f });
	const uint64_t one_group = a.collide_with(b, options);
	const auto one_stats = a.balancer_stats();
	options.use_load_balancer = false;
	const uint64_t wavefront = a.collide_with(b, options);
	EXPECT_EQ(small_stacks, wavefront);
	EXPECT_EQ(one_group, wavefront);
	EXPECT_GT(small_stats.balance_steps, 3u) << "tiny stacks must force several redistributions";
	EXPECT_GE(small_stats.kernel_launches, small_stats.balance_steps);
	EXPECT_GE(one_stats.balance_steps, 1u);
	std::cout << "[vulkan] balancer with 64 tasks x 64 items: " << small_stats.kernel_launches << " launches, " << small_stats.balance_steps
		<< " redistributions; with one group: " << one_stats.kernel_launches << " / " << one_stats.balance_steps << std::endl;
}

TEST_F(VulkanTest, BalancerStopsTheKernelWhenEnoughWorkGroupsAreIdle)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	// The idle counter only decides WHEN a work kernel ends, never what it computes. At the start one stack holds the single work
	// item and the other groups idle at once: with a low idle fraction the kernel ends after a few iterations and the work is spread
	// out, with 1.0 the first group works until its stack is full or empty.
	std::mt19937_64 rng(36);
	Octree a(ctx(), OctreeKind::Deterministic), b(ctx(), OctreeKind::Deterministic);
	a.build(to_voxels(random_voxels(rng, 3000, 60)));
	b.build(to_voxels(random_voxels(rng, 3000, 60)));
	OctreeTreeCollisionOptions options;
	options.mark_collisions = false;
	options.use_load_balancer = true;
	a.set_balancer_config({ 32, 700, 0.1f });
	const uint64_t eager = a.collide_with(b, options);
	const uint32_t eager_launches = a.balancer_stats().kernel_launches;
	a.set_balancer_config({ 32, 700, 1.0f });
	const uint64_t patient = a.collide_with(b, options);
	const uint32_t patient_launches = a.balancer_stats().kernel_launches;
	EXPECT_EQ(eager, patient);
	EXPECT_GT(eager_launches, patient_launches) << "a low idle fraction must end the work kernels earlier";
}

TEST_F(VulkanTest, BalancerReportsStacksThatAreTooSmall)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(35);
	Octree a(ctx(), OctreeKind::Deterministic), b(ctx(), OctreeKind::Deterministic);
	a.build(to_voxels(random_voxels(rng, 2000, 40)));
	b.build(to_voxels(random_voxels(rng, 2000, 40)));
	OctreeTreeCollisionOptions options;
	options.use_load_balancer = true;
	options.mark_collisions = false;
	a.set_balancer_config({ 1, 64, 1.0f }); // stack threshold 8 items on one stack: the pending items of two dense clusters need more than that
	EXPECT_THROW((void)a.collide_with(b, options), VulkanError);
	EXPECT_THROW(a.set_balancer_config({ 0, 700, 0.5f }), VulkanError);
	EXPECT_THROW(a.set_balancer_config({ 4, 63, 0.5f }), VulkanError);
	EXPECT_THROW(a.set_balancer_config({ 4, 700, 0.0f }), VulkanError);
	EXPECT_THROW(a.set_balancer_config({ 4, 700, 1.5f }), VulkanError);
	EXPECT_THROW(a.set_balancer_config({ 70000, 700, 0.5f }), VulkanError);
	a.set_balancer_config({}); // a valid configuration works again after the failure
	options.use_load_balancer = false;
	const uint64_t expected = a.collide_with(b, options);
	options.use_load_balancer = true;
	EXPECT_EQ(a.collide_with(b, options), expected);
	EXPECT_GT(expected, 0u);
}

TEST_F(VulkanTest, BalancerOnLargeTrees)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(34);
	const auto va = random_voxels(rng, 1'000'000, 200, 5000, 6000, 7000);
	const auto vb = random_voxels(rng, 1'000'000, 200, 5100, 6100, 7100);
	for (const auto kind : kKinds)
	{
		Octree a(ctx(), kind), b(ctx(), kind);
		a.build(to_voxels(va));
		b.build(to_voxels(vb));
		OctreeTreeCollisionOptions options;
		options.mark_collisions = false;
		const auto t0 = std::chrono::steady_clock::now();
		const uint64_t wavefront = a.collide_with(b, options);
		const auto t1 = std::chrono::steady_clock::now();
		options.use_load_balancer = true;
		const uint64_t balanced = a.collide_with(b, options);
		const auto t2 = std::chrono::steady_clock::now();
		const auto stats = a.balancer_stats();
		std::cout << "[vulkan] octree vs octree (" << name(kind) << "), 2 x ~1 M voxels: wavefront "
			<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms, load balancer " << std::chrono::duration<double, std::milli>(t2 - t1).count()
			<< " ms (" << stats.kernel_launches << " launches, " << stats.balance_steps << " redistributions, " << ctx().info().name << ")" << std::endl;
		EXPECT_EQ(balanced, wavefront);
		EXPECT_GT(balanced, 0u);

		OctreeExtractOptions extract;
		const auto e0 = std::chrono::steady_clock::now();
		const auto wave_cubes = a.extract(extract);
		const auto e1 = std::chrono::steady_clock::now();
		extract.use_load_balancer = true;
		const auto lb_cubes = a.extract(extract);
		const auto e2 = std::chrono::steady_clock::now();
		std::cout << "[vulkan] octree extract (" << name(kind) << "), ~1 M voxels: wavefront " << std::chrono::duration<double, std::milli>(e1 - e0).count()
			<< " ms, load balancer " << std::chrono::duration<double, std::milli>(e2 - e1).count() << " ms (" << wave_cubes.size() << " nodes)" << std::endl;
		EXPECT_EQ(lb_cubes.size(), wave_cubes.size());
		EXPECT_TRUE(sorted(lb_cubes) == sorted(wave_cubes));
	}
}
