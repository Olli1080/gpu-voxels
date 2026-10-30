// Buffer device addresses (S6a): the octree stores addresses of child blocks in its nodes, so shaders must be able to
// read and write memory that is NOT bound as a descriptor, across separately allocated buffers.
#include "vulkan_fixture.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;

namespace
{
	std::vector<uint32_t> load(const char* name)
	{
		return ComputePipeline::load_spirv(std::filesystem::path(GVL_TEST_SHADER_DIR) / (std::string(name) + ".spv"));
	}

	struct WritePush { uint64_t target; uint32_t count; uint32_t pad; };
	struct ChasePush { uint32_t count; uint32_t max_steps; };
	struct ChaseNode { uint64_t next; uint32_t value; uint32_t pad; };
	static_assert(sizeof(ChaseNode) == 16);

	constexpr uint32_t kSentinel = 0xDEADBEEFu;
}

TEST_F(VulkanTest, DeviceAddressIsReportedAndStable)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported by " << ctx().info().name;

	std::vector<DeviceBuffer> buffers;
	std::set<uint64_t> addresses;
	for (uint64_t size : { 4ull, 12ull, 100ull, 4096ull, 65536ull + 4, 1ull << 22 })
	{
		buffers.emplace_back(ctx(), size);
		const uint64_t address = buffers.back().device_address();
		EXPECT_NE(address, 0u);
		EXPECT_EQ(address % 8, 0u) << "size " << size; // the octree packs addresses of 8-byte aligned blocks (measured property of this driver)
		EXPECT_TRUE(addresses.insert(address).second) << "address of two live buffers is equal";
	}
	// buffers do not overlap
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	for (const auto& b : buffers)
		ranges.emplace_back(b.device_address(), b.device_address() + b.size());
	std::sort(ranges.begin(), ranges.end());
	for (size_t i = 1; i < ranges.size(); ++i)
		EXPECT_LE(ranges[i - 1].second, ranges[i].first);

	// the address survives moving the buffer object
	DeviceBuffer original(ctx(), 256);
	const uint64_t address = original.device_address();
	DeviceBuffer moved(std::move(original));
	EXPECT_EQ(moved.device_address(), address);
	EXPECT_THROW((void)original.device_address(), VulkanError); // the moved-from buffer has no address any more
	DeviceBuffer assigned(ctx(), 64);
	assigned = std::move(moved);
	EXPECT_EQ(assigned.device_address(), address);
	EXPECT_THROW((void)moved.device_address(), VulkanError); // moved-from buffer has no address any more
}

TEST_F(VulkanTest, ShaderWritesThroughDeviceAddressIncludingOffsets)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";

	constexpr uint32_t total = 1024;
	ComputePipeline pipeline(ctx(), load("bda_write.comp"), 1, sizeof(WritePush));
	for (const auto [offset_words, count] : { std::pair<uint32_t, uint32_t>{ 0, total }, { 0, 1 }, { 5, 100 }, { 1000, 24 }, { 3, 0 } })
	{
		DeviceBuffer target(ctx(), total * 4);
		DeviceBuffer written(ctx(), 4);
		target.upload<uint32_t>(std::vector<uint32_t>(total, kSentinel));
		written.upload<uint32_t>(std::vector<uint32_t>{ 0u });

		const WritePush push{ target.device_address() + uint64_t(offset_words) * 4, count, 0 };
		CommandBatch batch(ctx());
		batch.dispatch(pipeline, { &written }, std::max<uint32_t>(1, (count + 63) / 64), 1, 1, &push, sizeof(push));
		batch.submit_and_wait();

		const auto data = target.download_vector<uint32_t>(total);
		for (uint32_t i = 0; i < total; ++i)
		{
			const bool inside = i >= offset_words && i < offset_words + count;
			ASSERT_EQ(data[i], inside ? (i - offset_words) * 3 + 1 : kSentinel) << "offset " << offset_words << " count " << count << " word " << i;
		}
		EXPECT_EQ(written.download_vector<uint32_t>(1)[0], count);
	}
}

TEST_F(VulkanTest, ShaderFollowsPointersAcrossBuffers)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";

	constexpr uint32_t buffer_count = 5, nodes_per_buffer = 16, node_count = buffer_count * nodes_per_buffer;
	std::vector<DeviceBuffer> pool;
	for (uint32_t b = 0; b < buffer_count; ++b)
		pool.emplace_back(ctx(), nodes_per_buffer * sizeof(ChaseNode));
	auto address_of = [&](uint32_t node) { return pool[node / nodes_per_buffer].device_address() + uint64_t(node % nodes_per_buffer) * sizeof(ChaseNode); };

	// chains of different lengths over a random permutation of all nodes; node 0 is kept for the cyclic chain
	std::mt19937 rng(42);
	std::vector<uint32_t> order(node_count - 1);
	std::iota(order.begin(), order.end(), 1u);
	std::shuffle(order.begin(), order.end(), rng);
	std::vector<ChaseNode> nodes(node_count);
	for (uint32_t n = 0; n < node_count; ++n)
		nodes[n] = { 0, 1000 + n * 7, 0 };

	constexpr uint32_t chain_count = 12, max_steps = 10;
	std::vector<uint64_t> heads(chain_count, 0);
	std::vector<uint32_t> expected_sum(chain_count, 0), expected_steps(chain_count, 0);
	size_t cursor = 0;
	for (uint32_t c = 1; c + 1 < chain_count; ++c) // chain 0 is empty, the last one cyclic
	{
		const uint32_t length = c % 6 + 1;
		uint32_t previous = ~0u;
		for (uint32_t k = 0; k < length; ++k)
		{
			const uint32_t node = order[cursor++];
			if (k == 0)
				heads[c] = address_of(node);
			else
				nodes[previous].next = address_of(node);
			previous = node;
			expected_sum[c] += nodes[node].value;
			++expected_steps[c];
		}
	}
	nodes[0].next = address_of(0); // self loop
	heads[chain_count - 1] = address_of(0);
	expected_sum[chain_count - 1] = max_steps * nodes[0].value;
	expected_steps[chain_count - 1] = max_steps;

	for (uint32_t b = 0; b < buffer_count; ++b)
		pool[b].upload<ChaseNode>(std::vector<ChaseNode>(nodes.begin() + b * nodes_per_buffer, nodes.begin() + (b + 1) * nodes_per_buffer));

	std::vector<uint32_t> head_words;
	for (uint64_t h : heads)
	{
		head_words.push_back(uint32_t(h));
		head_words.push_back(uint32_t(h >> 32));
	}
	DeviceBuffer head_buffer(ctx(), head_words.size() * 4);
	DeviceBuffer result(ctx(), chain_count * 8);
	head_buffer.upload<uint32_t>(head_words);

	ComputePipeline pipeline(ctx(), load("bda_chase.comp"), 2, sizeof(ChasePush));
	CommandBatch batch(ctx());
	batch.dispatch(pipeline, { &head_buffer, &result }, 1, ChasePush{ chain_count, max_steps });
	batch.submit_and_wait();

	const auto r = result.download_vector<uint32_t>(chain_count * 2);
	for (uint32_t c = 0; c < chain_count; ++c)
	{
		EXPECT_EQ(r[2 * c], expected_sum[c]) << "chain " << c;
		EXPECT_EQ(r[2 * c + 1], expected_steps[c]) << "chain " << c;
	}
}

TEST_F(VulkanTest, DeviceAddressIsUnavailableWhenSwitchedOff)
{
	ContextCreateInfo info;
	info.require_validation = std::getenv("GVL_ALLOW_NO_VALIDATION") == nullptr;
	info.enable_buffer_device_address = false;
	Context context(info);
	EXPECT_FALSE(context.info().buffer_device_address);
	DeviceBuffer buffer(context, 1024); // ordinary buffers keep working
	buffer.upload<uint32_t>(std::vector<uint32_t>(256, 7));
	EXPECT_EQ(buffer.download_vector<uint32_t>(256)[255], 7u);
	EXPECT_THROW((void)buffer.device_address(), VulkanError);
}
