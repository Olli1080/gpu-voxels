// Slice S6d-4: timings of the dynamic octree (insert, propagate, node insert, rebuild, and what a long series of small inserts costs the queries).
// A measurement, not a test: disabled by default, run with
//   test_vulkan --gtest_also_run_disabled_tests --gtest_filter=VulkanTest.DISABLED_OctreeDynamicTimings
// For numbers that mean something use a Release build and no validation layer:
//   GVL_ALLOW_NO_VALIDATION=1 VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation test_vulkan ...
// GVL_TIMING_VOXELS (default 1000000) sets the size of the base tree. Every line is "[timing] ..."; the minimum of 3 runs is printed.
#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;

namespace
{
	using Clock = std::chrono::steady_clock;

	template<typename F>
	double milliseconds(F&& f)
	{
		const auto t0 = Clock::now();
		f();
		return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
	}

	std::vector<OctreeVoxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t extent)
	{
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::vector<OctreeVoxel> v;
		v.reserve(count);
		for (size_t i = 0; i < count; ++i)
			v.push_back({ d(rng), d(rng), d(rng) });
		return v;
	}

	struct Workload
	{
		const char* name;
		uint32_t extent;
	};

	size_t base_voxels()
	{
		const char* env = std::getenv("GVL_TIMING_VOXELS");
		return env ? static_cast<size_t>(std::strtoull(env, nullptr, 10)) : 1000000;
	}

	std::string mib(uint64_t bytes) { return std::to_string(bytes / (1024 * 1024)) + " MiB"; }
}

TEST_F(VulkanTest, DISABLED_OctreeDynamicTimings)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const size_t n = base_voxels();
	const Workload workloads[] = { { "random 12%-ish dense", 200 }, { "random sparse 2048^3", 2048 } };
	constexpr int kRepeats = 3;
	std::cout << "[timing] device " << ctx().info().name << ", validation " << (ctx().info().validation_enabled ? "ON" : "off") << ", base tree " << n << " voxels" << std::endl;

	for (const auto kind : { OctreeKind::Deterministic, OctreeKind::Probabilistic })
	{
		for (const Workload& w : workloads)
		{
			const std::string tag = std::string(kind == OctreeKind::Probabilistic ? "prob" : "det ") + " " + w.name;
			std::mt19937_64 rng(42);
			const auto base = random_voxels(rng, n, w.extent);
			auto fresh = [&]
			{
				auto t = std::make_unique<Octree>(ctx(), kind);
				if (const char* e = std::getenv("GVL_TIMING_EXACT_LIST_THRESHOLD")) // A/B of the exact propagate path: bytes, 0 = always count first
					t->set_exact_list_threshold(std::strtoull(e, nullptr, 10));
				t->build(base);
				return t;
			};
			auto best = [&](auto&& prepare_and_run) // prepare_and_run() returns the time of the operation in ms
			{
				double m = 1e300;
				for (int i = 0; i < kRepeats; ++i)
					m = std::min<double>(m, prepare_and_run());
				return m;
			};

			const double build = best([&] { Octree t(ctx(), kind); return milliseconds([&] { t.build(base); }); });
			auto reference = fresh();
			const OctreeStats stats = reference->memory_usage();
			std::cout << "[timing] " << tag << ": build " << build << " ms, " << mib(stats.bytes) << ", " << stats.nodes() << " nodes" << std::endl;

			// insert of a batch followed by propagate, separately timed (batch of new random voxels in the same volume)
			for (const size_t batch : { size_t(1), size_t(100), size_t(10000), size_t(100000), n })
			{
				const auto voxels = random_voxels(rng, batch, w.extent);
				double insert_ms = 1e300, propagate_ms = 1e300;
				for (int i = 0; i < kRepeats; ++i)
				{
					auto t = fresh();
					insert_ms = std::min(insert_ms, milliseconds([&] { t->insert(voxels, OctreeDatum::Occupied, false); }));
					propagate_ms = std::min(propagate_ms, milliseconds([&] { t->propagate(); }));
				}
				std::cout << "[timing] " << tag << ": insert " << batch << " voxels " << insert_ms << " ms + propagate " << propagate_ms << " ms" << std::endl;
			}

			{
				// the host part of an insert is sorting and deduplicating the ids: how much of the largest batch is that?
				std::vector<uint64_t> ids(n);
				for (auto& id : ids)
					id = rng() % (uint64_t(1) << 42);
				const double sort_ms = milliseconds([&] { std::sort(ids.begin(), ids.end()); ids.erase(std::unique(ids.begin(), ids.end()), ids.end()); });
				std::cout << "[timing] " << tag << ": host std::sort + unique of " << n << " ids " << sort_ms << " ms" << std::endl;
			}

			// free space as nodes of level 3 (8^3 voxels each)
			{
				std::vector<uint64_t> ids;
				std::uniform_int_distribution<uint64_t> d(0, (uint64_t(1) << 33) - 1);
				for (int i = 0; i < 10000; ++i)
					ids.push_back(d(rng) % (uint64_t(1) << (3 * (14 - 3))));
				const double t_ms = best([&] { auto t = fresh(); return milliseconds([&] { t->insert_nodes(3, ids, OctreeDatum::Free, OctreeInsertMode::Hard); }); });
				std::cout << "[timing] " << tag << ": insert_nodes level 3, 10000 nodes incl. propagate " << t_ms << " ms" << std::endl;
			}

			// a long series of small inserts: cost per insert, growth, and what it does to the queries; then rebuild
			{
				auto t = fresh();
				constexpr int kSteps = 200;
				constexpr size_t kBatch = 1000;
				std::vector<double> per_insert;
				for (int i = 0; i < kSteps; ++i)
				{
					const auto voxels = random_voxels(rng, kBatch, w.extent);
					per_insert.push_back(milliseconds([&] { t->insert(voxels); }));
				}
				std::sort(per_insert.begin(), per_insert.end());
				const OctreeStats grown = t->memory_usage();
				std::cout << "[timing] " << tag << ": " << kSteps << " inserts of " << kBatch << " voxels with propagate: median " << per_insert[kSteps / 2] << " ms, max "
					<< per_insert.back() << " ms; tree " << mib(grown.bytes) << ", " << grown.chunks << " chunks" << std::endl;
				uint64_t garbage = 0;
				for (const auto g : grown.garbage_blocks)
					garbage += g;
				std::vector<uint64_t> probes(100000);
				std::uniform_int_distribution<uint64_t> d(0, (uint64_t(w.extent) * w.extent * w.extent) - 1);
				for (auto& p : probes)
					p = d(rng);
				const double find_before = best([&] { return milliseconds([&] { (void)t->find(probes); }); });
				const double extract_before = best([&] { return milliseconds([&] { (void)t->extract(); }); });
				const double usage_before = best([&] { return milliseconds([&] { (void)t->memory_usage(); }); });
				const double rebuild = milliseconds([&] { t->rebuild(); });
				const OctreeStats packed = t->memory_usage();
				const double find_after = best([&] { return milliseconds([&] { (void)t->find(probes); }); });
				const double extract_after = best([&] { return milliseconds([&] { (void)t->extract(); }); });
				const double usage_after = best([&] { return milliseconds([&] { (void)t->memory_usage(); }); });
				std::cout << "[timing] " << tag << ": garbage blocks before rebuild " << garbage << ", rebuild " << rebuild << " ms, tree " << mib(grown.bytes) << " -> " << mib(packed.bytes)
					<< ", chunks " << grown.chunks << " -> " << packed.chunks << std::endl;
				std::cout << "[timing] " << tag << ": find 100000 ids " << find_before << " -> " << find_after << " ms, extract " << extract_before << " -> " << extract_after
					<< " ms, memory_usage() " << usage_before << " -> " << usage_after << " ms" << std::endl;
				EXPECT_NO_THROW(t->check_tree());
			}
		}

		// a dense cube is made of uniform nodes; free voxels inside it replace their child blocks and leave garbage: what rebuild gives back
		{
			const std::string tag = std::string(kind == OctreeKind::Probabilistic ? "prob" : "det ") + " dense cube 128^3";
			std::vector<OctreeVoxel> cube;
			cube.reserve(128 * 128 * 128);
			for (uint32_t x = 0; x < 128; ++x)
				for (uint32_t y = 0; y < 128; ++y)
					for (uint32_t z = 0; z < 128; ++z)
						cube.push_back({ x, y, z });
			Octree t(ctx(), kind);
			t.build(cube);
			const OctreeStats built = t.memory_usage();
			std::mt19937_64 rng(7);
			constexpr int kSteps = 100;
			std::vector<double> per_insert;
			for (int i = 0; i < kSteps; ++i)
			{
				const auto voxels = random_voxels(rng, 1000, 128);
				per_insert.push_back(milliseconds([&] { t.insert(voxels, OctreeDatum::Free); }));
			}
			std::sort(per_insert.begin(), per_insert.end());
			const double usage = milliseconds([&] { (void)t.memory_usage(); });
			const OctreeStats grown = t.memory_usage();
			uint64_t garbage = 0;
			for (const auto g : grown.garbage_blocks)
				garbage += g;
			const double needs = milliseconds([&] { OctreeRebuildPolicy p; p.max_bytes = 0; p.max_garbage_fraction = 0.5; t.set_rebuild_policy(p); (void)t.needs_rebuild(); });
			const double rebuild = milliseconds([&] { t.rebuild(); });
			const OctreeStats packed = t.memory_usage();
			std::cout << "[timing] " << tag << ": built " << mib(built.bytes) << ", " << kSteps << " inserts of 1000 free voxels: median " << per_insert[kSteps / 2] << " ms; tree " << mib(grown.bytes)
				<< ", garbage blocks " << garbage << ", chunks " << grown.chunks << "; memory_usage() " << usage << " ms, needs_rebuild(fraction) " << needs << " ms; rebuild " << rebuild
				<< " ms -> " << mib(packed.bytes) << ", chunks " << packed.chunks << std::endl;
			EXPECT_NO_THROW(t.check_tree());
		}
	}
}
