// Octree checkpoint (between S6c and S6d): the measurements listed in Analysis/2026-10-29_octree_design_review/design_options.md
//  * sort of 64-bit keys (Morton ids have 42 bit) at 4 M and 16 M elements, against the 32-bit sort,
//  * node counts and bytes per voxel of the faithful tree for dense and sparse inputs, against what the index based pool
//    (Option 2) and the sorted level arrays (Option 3) would need for the same nodes (computed, not measured),
//  * frontier sizes (work items per level) of tree vs tree, from the independent CPU recursion.
// The numbers are printed ("[vulkan] ..."); the assertions check what can be checked independently of the hardware.
#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/Primitives.h>

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
	double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

	std::vector<OctreeVoxel> to_voxels(const std::vector<ref::Voxel>& v)
	{
		std::vector<OctreeVoxel> out;
		out.reserve(v.size());
		for (const auto& x : v)
			out.push_back({ x.x, x.y, x.z });
		return out;
	}

	std::vector<ref::Voxel> dense_cube(uint32_t side, uint32_t ox, uint32_t oy, uint32_t oz)
	{
		std::vector<ref::Voxel> v;
		v.reserve(size_t(side) * side * side);
		for (uint32_t z = 0; z < side; ++z)
			for (uint32_t y = 0; y < side; ++y)
				for (uint32_t x = 0; x < side; ++x)
					v.push_back({ ox + x, oy + y, oz + z });
		return v;
	}

	/// Voxels whose centre has a distance in [radius, radius + 1) from the centre point: a one voxel thick surface.
	std::vector<ref::Voxel> sphere_shell(uint32_t cx, uint32_t cy, uint32_t cz, uint32_t radius)
	{
		std::vector<ref::Voxel> v;
		const int r = int(radius) + 1;
		for (int z = -r; z <= r; ++z)
			for (int y = -r; y <= r; ++y)
				for (int x = -r; x <= r; ++x)
				{
					const double d = std::sqrt(double(x * x + y * y + z * z));
					if (d >= double(radius) && d < double(radius) + 1.0)
						v.push_back({ uint32_t(int(cx) + x), uint32_t(int(cy) + y), uint32_t(int(cz) + z) });
				}
		return v;
	}

	std::vector<ref::Voxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent, uint32_t ox, uint32_t oy, uint32_t oz)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<ref::Voxel> v;
		v.reserve(count);
		for (size_t i = 0; i < count; ++i)
			v.push_back({ ox + d(rng), oy + d(rng), oz + d(rng) });
		return v;
	}

	struct Workload
	{
		std::string name;
		std::vector<ref::Voxel> voxels;
	};
}

TEST_F(VulkanTest, CheckpointSortOf64BitKeys)
{
	Primitives primitives(ctx());
	std::mt19937_64 rng(2026);
	// the sort is bound by 65 535 work groups of 256 elements (Primitives.cpp), a little below the nominal 2^24
	constexpr uint32_t kSortLimit = 65535u * 256u;
	for (const uint32_t n : { 1u << 22, kSortLimit })
	{
		std::vector<uint64_t> keys(n);
		for (auto& k : keys)
			k = rng() & ((uint64_t(1) << 42) - 1);
		std::vector<uint32_t> values(n);
		std::iota(values.begin(), values.end(), 0u);
		std::vector<uint32_t> keys32(n);
		for (uint32_t i = 0; i < n; ++i)
			keys32[i] = uint32_t(keys[i]);
		std::vector<uint32_t> values32 = values;
		const auto original = keys;

		const auto t0 = std::chrono::steady_clock::now();
		primitives.sort_pairs64(keys, values, 42);
		const auto t1 = std::chrono::steady_clock::now();
		primitives.sort_pairs(keys32, values32, 32);
		const auto t2 = std::chrono::steady_clock::now();
		std::cout << "[vulkan] sort of " << (double(n) / 1048576.0) << " M pairs incl. transfers: 64-bit keys (42 bit, 6 passes) " << ms(t0, t1)
			<< " ms, 32-bit keys (32 bit, 4 passes) " << ms(t1, t2) << " ms (" << ctx().info().name << ")" << std::endl;

		EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
		EXPECT_TRUE(std::is_sorted(keys32.begin(), keys32.end()));
		for (size_t i = 0; i < keys.size(); i += 4099)
			ASSERT_EQ(original[values[i]], keys[i]) << "value must still belong to its key";
	}
	std::vector<uint64_t> too_many(kSortLimit + 1, 0);
	std::vector<uint32_t> too_many_values(kSortLimit + 1, 0);
	EXPECT_THROW(primitives.sort_pairs64(too_many, too_many_values, 42), VulkanError);
}

TEST_F(VulkanTest, CheckpointBytesPerVoxel)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(2027);
	std::vector<Workload> workloads;
	workloads.push_back({ "dense cube 128^3", dense_cube(128, 4000, 5000, 6000) });
	workloads.push_back({ "sphere shell r=300", sphere_shell(8000, 8000, 8000, 300) });
	workloads.push_back({ "random 1 M in 200^3 (12 %)", random_voxels(rng, 1'000'000, 200, 3000, 3000, 3000) });
	workloads.push_back({ "random 1 M in 2048^3 (0.01 %)", random_voxels(rng, 1'000'000, 2048, 1000, 1000, 1000) });
	workloads.push_back({ "random 1 M in 16384^3 (sparse)", random_voxels(rng, 1'000'000, 16384, 0, 0, 0) });

	for (const auto& workload : workloads)
		for (const auto kind : kKinds)
		{
			Octree tree(ctx(), kind);
			const auto t0 = std::chrono::steady_clock::now();
			tree.build(to_voxels(workload.voxels));
			const auto t1 = std::chrono::steady_clock::now();
			const OctreeStats stats = tree.memory_usage();
			const uint64_t voxels = tree.voxel_count();
			ASSERT_GT(voxels, 0u);
			const auto nodes = tree.dump();
			const uint64_t real_nodes = uint64_t(std::count_if(nodes.begin(), nodes.end(), [&](const OctreeNode& n) { return kind == OctreeKind::Probabilistic ? n.occupancy != fmt::kProbUnknown : (n.status & (fmt::kFree | fmt::kOccupied)) != 0; }));
			const bool prob = kind == OctreeKind::Probabilistic;

			// bytes the faithful tree must occupy: blocks of 8 nodes per level, leaves 1 / 2 bytes, inner nodes 8 / 12 bytes
			const uint64_t inner_bytes = prob ? 12 : 8, leaf_bytes = prob ? 2 : 1;
			uint64_t expected_bytes = stats.blocks[0] * 8 * leaf_bytes;
			for (uint32_t level = 1; level < fmt::kLevelCount; ++level)
				expected_bytes += stats.blocks[level] * 8 * inner_bytes;
			EXPECT_EQ(stats.bytes, expected_bytes) << workload.name << " " << name(kind);

			// Option 2 (index pool): 8 bytes per inner node, leaves as above (fillers included); Option 3: key (8 bytes) + status (+ occupancy) for
			// every real node, i.e. every node with a known state (free / occupied, probabilistic: occupancy known); the unknown fillers of the blocks are not stored
			uint64_t option2 = stats.blocks[0] * 8 * leaf_bytes;
			for (uint32_t level = 1; level < fmt::kLevelCount; ++level)
				option2 += stats.blocks[level] * 8 * 8;
			const uint64_t option3 = real_nodes * (prob ? 10 : 9);
			std::cout << "[vulkan] bytes per voxel, " << workload.name << " (" << name(kind) << "): " << voxels << " voxels, " << stats.nodes()
				<< " allocated nodes (" << real_nodes << " known), build " << ms(t0, t1) << " ms; faithful " << stats.bytes << " B = "
				<< double(stats.bytes) / double(voxels) << " B/voxel, index pool " << double(option2) / double(voxels) << " B/voxel, sorted levels "
				<< double(option3) / double(voxels) << " B/voxel" << std::endl;
			EXPECT_EQ(nodes.size(), stats.nodes() - 7) << "every block is on a path; only the root block has unused nodes";
			EXPECT_LE(real_nodes, nodes.size());
			EXPECT_NO_THROW(tree.check_tree());
		}
}

TEST_F(VulkanTest, CheckpointFrontierOfTreeCollision)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(2028);
	struct Pair
	{
		std::string name;
		std::vector<ref::Voxel> a, b;
	};
	std::vector<Pair> pairs;
	pairs.push_back({ "two spheres r=150, centres 100 apart", sphere_shell(5000, 6000, 7000, 150), sphere_shell(5100, 6000, 7000, 150) });
	pairs.push_back({ "two random clusters 150 k in 200^3", random_voxels(rng, 150'000, 200, 5000, 6000, 7000), random_voxels(rng, 150'000, 200, 5100, 6100, 7100) });
	pairs.push_back({ "two random sets 150 k in 4096^3", random_voxels(rng, 150'000, 4096, 2000, 2000, 2000), random_voxels(rng, 150'000, 4096, 2500, 2500, 2500) });

	for (const auto& pair : pairs)
		for (const auto kind : kKinds)
		{
			const ref::OctreeReference ra(to_ref(kind), pair.a), rb(to_ref(kind), pair.b);
			const auto expected = ra.collide_tree(rb, 0, 10, false);
			Octree a(ctx(), kind), b(ctx(), kind);
			a.build(to_voxels(pair.a));
			b.build(to_voxels(pair.b));
			OctreeTreeCollisionOptions options;
			options.mark_collisions = false;
			EXPECT_EQ(a.collide_with(b, options), expected.voxels) << pair.name << " " << name(kind);

			// work items (pairs of blocks) per level of a level-synchronous traversal, from the independent recursion
			const OctreeStats sa = a.memory_usage();
			std::string line;
			uint64_t peak = 0, total = 0;
			for (uint32_t level = 13; level >= 1; --level)
			{
				const uint64_t items = expected.block_pairs[level];
				peak = std::max(peak, items);
				total += items;
				line += " L" + std::to_string(level) + "=" + std::to_string(items);
			}
			std::cout << "[vulkan] frontier of " << pair.name << " (" << name(kind) << "): " << expected.voxels << " voxels in collision, peak "
				<< peak << " block pairs, total " << total << " (a: " << a.voxel_count() << " voxels, " << sa.nodes() << " nodes);" << line << std::endl;
			EXPECT_GT(peak, 0u);
			EXPECT_LE(peak, std::max(sa.nodes(), b.memory_usage().nodes()) / 8);
		}
}
