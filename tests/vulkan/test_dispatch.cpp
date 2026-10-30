#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;

namespace
{
	struct CountPush { uint32_t count; };
	struct ScalePush { uint32_t count; uint32_t factor; };
	constexpr uint32_t kLocalSize = 64;

	std::vector<uint32_t> load(const char* name)
	{
		return ComputePipeline::load_spirv(std::filesystem::path(GVL_TEST_SHADER_DIR) / (std::string(name) + ".spv"));
	}

	uint32_t groups_for(uint32_t count) { return (count + kLocalSize - 1) / kLocalSize; }
}

TEST_F(VulkanTest, DispatchAddOne)
{
	constexpr uint32_t n = 1'000'003; // not a multiple of the work group size
	std::vector<uint32_t> data(n);
	std::iota(data.begin(), data.end(), 0u);
	DeviceBuffer buffer(ctx(), n * sizeof(uint32_t));
	buffer.upload<uint32_t>(data);

	ComputePipeline add_one(ctx(), load("add_one.comp"), 1, sizeof(CountPush));
	CommandBatch batch(ctx());
	batch.dispatch(add_one, { &buffer }, groups_for(n), CountPush{ n });
	const auto t0 = std::chrono::steady_clock::now();
	batch.submit_and_wait();
	const auto t1 = std::chrono::steady_clock::now();
	std::cout << "[vulkan] add_one over " << n << " elements: "
		<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms (" << ctx().info().name << ")" << std::endl;

	const auto result = buffer.download_vector<uint32_t>(n);
	for (uint32_t i = 0; i < n; ++i)
		ASSERT_EQ(result[i], i + 1) << "index " << i;
}

TEST_F(VulkanTest, DependentDispatchesWithBarrierInOneSubmit)
{
	constexpr uint32_t n = 300'000;
	std::vector<uint32_t> data(n);
	std::iota(data.begin(), data.end(), 10u);
	DeviceBuffer in(ctx(), n * sizeof(uint32_t));
	DeviceBuffer out(ctx(), n * sizeof(uint32_t));
	in.upload<uint32_t>(data);
	out.fill(0);

	ComputePipeline add_one(ctx(), load("add_one.comp"), 1, sizeof(CountPush));
	ComputePipeline scale(ctx(), load("scale_copy.comp"), 2, sizeof(ScalePush));
	CommandBatch batch(ctx());
	batch.dispatch(add_one, { &in }, groups_for(n), CountPush{ n });
	batch.barrier(); // scale reads what add_one wrote
	batch.dispatch(scale, { &in, &out }, groups_for(n), ScalePush{ n, 2 });
	EXPECT_EQ(batch.dispatch_count(), 2u);
	batch.submit_and_wait();

	const auto result = out.download_vector<uint32_t>(n);
	for (uint32_t i = 0; i < n; ++i)
		ASSERT_EQ(result[i], (data[i] + 1) * 2) << "index " << i;
}

// More dispatches than descriptor sets per pool (64): the batch must allocate additional pools.
TEST_F(VulkanTest, ManyDispatchesInOneBatch)
{
	constexpr uint32_t n = 4096;
	constexpr uint32_t dispatches = 200;
	DeviceBuffer buffer(ctx(), n * sizeof(uint32_t));
	buffer.fill(0);
	ComputePipeline add_one(ctx(), load("add_one.comp"), 1, sizeof(CountPush));

	CommandBatch batch(ctx());
	for (uint32_t d = 0; d < dispatches; ++d)
	{
		batch.dispatch(add_one, { &buffer }, groups_for(n), CountPush{ n });
		batch.barrier();
	}
	batch.submit_and_wait();

	const auto result = buffer.download_vector<uint32_t>(n);
	EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](uint32_t v) { return v == dispatches; }));
}

TEST_F(VulkanTest, PipelineIsReusableAcrossBatches)
{
	constexpr uint32_t n = 1000;
	DeviceBuffer buffer(ctx(), n * sizeof(uint32_t));
	buffer.fill(0);
	ComputePipeline add_one(ctx(), load("add_one.comp"), 1, sizeof(CountPush));
	for (int round = 1; round <= 3; ++round)
	{
		CommandBatch batch(ctx());
		batch.dispatch(add_one, { &buffer }, groups_for(n), CountPush{ n });
		batch.submit_and_wait();
		EXPECT_EQ(buffer.download_vector<uint32_t>(1).front(), static_cast<uint32_t>(round));
	}
}

TEST_F(VulkanTest, Int64ArithmeticInShaders)
{
	constexpr uint32_t n = 1000;
	std::vector<uint32_t> hi(n), lo(n);
	for (uint32_t i = 0; i < n; ++i)
	{
		hi[i] = 0x80000000u + i * 12345u;
		lo[i] = 0xFFFFFFF0u + i;
	}
	DeviceBuffer hi_buffer(ctx(), n * 4), lo_buffer(ctx(), n * 4), result_buffer(ctx(), n * 8);
	hi_buffer.upload<uint32_t>(hi);
	lo_buffer.upload<uint32_t>(lo);

	ComputePipeline pipeline(ctx(), load("int64_mix.comp"), 3, sizeof(CountPush));
	CommandBatch batch(ctx());
	batch.dispatch(pipeline, { &hi_buffer, &lo_buffer, &result_buffer }, groups_for(n), CountPush{ n });
	batch.submit_and_wait();

	const auto result = result_buffer.download_vector<uint32_t>(2 * n);
	for (uint32_t i = 0; i < n; ++i)
	{
		const uint64_t expected = ((static_cast<uint64_t>(hi[i]) << 32) | lo[i]) * 3ull;
		const uint64_t actual = (static_cast<uint64_t>(result[2 * i + 1]) << 32) | result[2 * i];
		ASSERT_EQ(actual, expected) << "index " << i;
	}
}

TEST_F(VulkanTest, CommandBatchRejectsInvalidUse)
{
	DeviceBuffer buffer(ctx(), 256);
	ComputePipeline add_one(ctx(), load("add_one.comp"), 1, sizeof(CountPush));
	const CountPush push{ 64 };

	CommandBatch batch(ctx());
	EXPECT_THROW(batch.dispatch(add_one, { &buffer, &buffer }, 1, 1, 1, &push, sizeof(push)), VulkanError); // buffer count
	EXPECT_THROW(batch.dispatch(add_one, { &buffer }, 1, 1, 1, nullptr, 0), VulkanError);                   // push size
	EXPECT_THROW(batch.dispatch(add_one, { nullptr }, 1, 1, 1, &push, sizeof(push)), VulkanError);          // null buffer
	const uint32_t max_x = ctx().limits().maxComputeWorkGroupCount[0];
	if (max_x < UINT32_MAX)
		EXPECT_THROW(batch.dispatch(add_one, { &buffer }, max_x + 1, 1, 1, &push, sizeof(push)), VulkanError);
	batch.submit_and_wait(); // an empty batch is valid
	EXPECT_THROW(batch.submit_and_wait(), VulkanError);
	EXPECT_THROW(batch.barrier(), VulkanError);

	EXPECT_THROW(ComputePipeline(ctx(), load("add_one.comp"), ComputePipeline::kMaxStorageBuffers + 1), VulkanError);
	EXPECT_THROW(ComputePipeline(ctx(), load("add_one.comp"), 1, 3), VulkanError); // push size not a multiple of 4
	EXPECT_THROW(ComputePipeline(ctx(), std::vector<uint32_t>{}, 1, 0), VulkanError); // empty SPIR-V
}
