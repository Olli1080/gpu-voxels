// Morton codes on the GPU (S6a): morton.glsl must agree bit for bit with the host header and with the bit-loop reference.
#include "vulkan_fixture.h"

#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/Morton.h>

#include "../reference/octree_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference::octree;
namespace mo = gpu_voxels::vulkan::morton;

namespace
{
	struct CountPush { uint32_t count; };
	struct Case { uint32_t x, y, z, level; uint32_t x2, y2, z2, unused; };

	void run_and_check(Context& ctx, const std::vector<Case>& cases)
	{
		const uint32_t n = uint32_t(cases.size());
		std::vector<uint32_t> first, second;
		for (const auto& c : cases)
		{
			first.insert(first.end(), { c.x, c.y, c.z, c.level });
			second.insert(second.end(), { c.x2, c.y2, c.z2, 0u });
		}
		DeviceBuffer in1(ctx, n * 16ull), in2(ctx, n * 16ull), code(ctx, n * 8ull), decoded(ctx, n * 16ull), misc(ctx, n * 16ull);
		in1.upload<uint32_t>(first);
		in2.upload<uint32_t>(second);

		ComputePipeline pipeline(ctx, ComputePipeline::load_spirv(std::filesystem::path(GVL_TEST_SHADER_DIR) / "morton_check.comp.spv"), 5, sizeof(CountPush));
		CommandBatch batch(ctx);
		batch.dispatch(pipeline, { &in1, &in2, &code, &decoded, &misc }, (n + 63) / 64, CountPush{ n });
		batch.submit_and_wait();

		const auto c = code.download_vector<uint32_t>(n * 2ull);
		const auto d = decoded.download_vector<uint32_t>(n * 4ull);
		const auto m = misc.download_vector<uint32_t>(n * 4ull);
		for (uint32_t i = 0; i < n; ++i)
		{
			const Case& k = cases[i];
			const uint64_t expected = ref::morton(k.x, k.y, k.z);
			const uint64_t other = ref::morton(k.x2, k.y2, k.z2);
			const uint64_t gpu_code = (uint64_t(c[2 * i + 1]) << 32) | c[2 * i];
			ASSERT_EQ(gpu_code, expected) << "case " << i << " (" << k.x << "," << k.y << "," << k.z << ")";
			ASSERT_EQ(gpu_code, mo::code60(k.x, k.y, k.z));
			const auto back = ref::unmorton(expected);
			ASSERT_TRUE(d[4 * i] == back.x && d[4 * i + 1] == back.y && d[4 * i + 2] == back.z) << "case " << i;
			ASSERT_EQ(d[4 * i + 3], 0u);
			const uint64_t prefix = (uint64_t(m[4 * i + 1]) << 32) | m[4 * i];
			ASSERT_EQ(prefix, mo::z_order_prefix(expected, k.level)) << "case " << i << " level " << k.level;
			ASSERT_EQ(m[4 * i + 2], mo::z_order_node_id(expected, k.level)) << "case " << i << " level " << k.level;
			ASSERT_EQ(m[4 * i + 3], ref::common_level(expected, other)) << "case " << i << " " << expected << " vs " << other;
		}
	}
}

TEST_F(VulkanTest, MortonGlslMatchesHostAndReferenceOnBoundaries)
{
	const uint32_t edge[] = { 0, 1, 2, 3, 7, 1023, 1024, 1025, 16383, 16384, 524288, (1u << 20) - 1 };
	std::vector<Case> cases;
	uint32_t level = 0;
	for (uint32_t x : edge)
		for (uint32_t y : edge)
			for (uint32_t z : edge)
			{
				cases.push_back({ x, y, z, level % 15, z, x, y, 0 });
				++level;
			}
	run_and_check(ctx(), cases);
}

TEST_F(VulkanTest, MortonGlslMatchesHostAndReferenceOnRandomValues)
{
	std::mt19937_64 rng(99);
	std::vector<Case> cases;
	for (int i = 0; i < 50000; ++i)
	{
		Case c{};
		c.x = uint32_t(rng()) & 0xFFFFF;
		c.y = uint32_t(rng()) & 0xFFFFF;
		c.z = uint32_t(rng()) & 0xFFFFF;
		c.level = uint32_t(rng() % 15);
		// second voxel shares a random number of high digits with the first one, so every common level occurs
		const uint64_t a = ref::morton(c.x, c.y, c.z);
		const uint32_t shared_digits = uint32_t(rng() % 21);
		const uint64_t low_mask = shared_digits == 20 ? 0 : ((uint64_t(1) << (3 * (20 - shared_digits))) - 1);
		const auto b = ref::unmorton((a & ~low_mask) | (rng() & low_mask));
		c.x2 = b.x;
		c.y2 = b.y;
		c.z2 = b.z;
		cases.push_back(c);
	}
	// bits above 20 are ignored by the GPU code exactly like on the host
	for (int i = 0; i < 2000; ++i)
	{
		Case c{};
		c.x = uint32_t(rng());
		c.y = uint32_t(rng());
		c.z = uint32_t(rng());
		c.level = uint32_t(rng() % 15);
		c.x2 = uint32_t(rng());
		c.y2 = uint32_t(rng());
		c.z2 = uint32_t(rng());
		cases.push_back(c);
	}
	run_and_check(ctx(), cases);
}
