#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include <gpu_voxels/vulkan/DeviceBuffer.h>

using gpu_voxels::vulkan::DeviceBuffer;
using gpu_voxels::vulkan::VulkanError;
using gvl_test::VulkanTest;

TEST_F(VulkanTest, BufferRoundTrip)
{
	std::vector<uint32_t> in(1000);
	std::iota(in.begin(), in.end(), 7u);
	DeviceBuffer buffer(ctx(), in.size() * sizeof(uint32_t));
	buffer.upload<uint32_t>(in);
	EXPECT_EQ(buffer.download_vector<uint32_t>(in.size()), in);
}

TEST_F(VulkanTest, BufferUploadWithOffsetLeavesRestUntouched)
{
	DeviceBuffer buffer(ctx(), 4096);
	buffer.fill(0xFFFFFFFFu);
	const std::vector<uint32_t> patch{ 1, 2, 3, 4 };
	buffer.upload<uint32_t>(patch, 100 * sizeof(uint32_t));

	const auto all = buffer.download_vector<uint32_t>(1024);
	for (std::size_t i = 0; i < all.size(); ++i)
	{
		const bool in_patch = i >= 100 && i < 104;
		EXPECT_EQ(all[i], in_patch ? static_cast<uint32_t>(i - 99) : 0xFFFFFFFFu) << "index " << i;
	}
	EXPECT_EQ(buffer.download_vector<uint32_t>(4, 100 * sizeof(uint32_t)), patch);
}

TEST_F(VulkanTest, BufferFill)
{
	DeviceBuffer buffer(ctx(), 1 << 16);
	buffer.fill(0xA5A5A5A5u);
	const auto words = buffer.download_vector<uint32_t>((1 << 16) / 4);
	EXPECT_TRUE(std::all_of(words.begin(), words.end(), [](uint32_t w) { return w == 0xA5A5A5A5u; }));
}

TEST_F(VulkanTest, BufferRejectsInvalidSizesAndRanges)
{
	EXPECT_THROW(DeviceBuffer(ctx(), 0), VulkanError);
	EXPECT_THROW(DeviceBuffer(ctx(), ctx().limits().maxStorageBufferRange + 1), VulkanError);

	DeviceBuffer buffer(ctx(), 64);
	std::vector<uint8_t> data(128);
	EXPECT_THROW(buffer.upload(data.data(), 128), VulkanError);
	EXPECT_THROW(buffer.download(data.data(), 32, 48), VulkanError);
	DeviceBuffer odd(ctx(), 63);
	EXPECT_THROW(odd.fill(0), VulkanError); // not a multiple of 4
}

TEST_F(VulkanTest, BufferMoveTransfersOwnership)
{
	DeviceBuffer a(ctx(), 256);
	a.fill(5);
	DeviceBuffer b(std::move(a));
	EXPECT_EQ(b.size(), 256u);
	EXPECT_EQ(a.size(), 0u);
	EXPECT_EQ(b.download_vector<uint32_t>(1).front(), 5u);
	DeviceBuffer c(ctx(), 16);
	c = std::move(b);
	EXPECT_EQ(c.download_vector<uint32_t>(1).front(), 5u);
}

// More than the guaranteed minimum of maxStorageBufferRange (128 MiB) and more than two staging chunks.
TEST_F(VulkanTest, LargeBufferRoundTripAcrossStagingChunks)
{
	constexpr uint64_t bytes = 160ull * 1024 * 1024;
	if (ctx().limits().maxStorageBufferRange < bytes || ctx().info().device_local_heap_bytes < 2 * bytes)
		GTEST_SKIP() << "device too small for a " << (bytes >> 20) << " MiB buffer";

	std::vector<uint32_t> in(bytes / sizeof(uint32_t));
	for (std::size_t i = 0; i < in.size(); ++i)
		in[i] = static_cast<uint32_t>(i) * 2654435761u;

	DeviceBuffer buffer(ctx(), bytes);
	const auto t0 = std::chrono::steady_clock::now();
	buffer.upload<uint32_t>(in);
	const auto t1 = std::chrono::steady_clock::now();
	const auto out = buffer.download_vector<uint32_t>(in.size());
	const auto t2 = std::chrono::steady_clock::now();

	const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
	std::cout << "[vulkan] " << (bytes >> 20) << " MiB upload " << ms(t0, t1) << " ms, download " << ms(t1, t2)
		<< " ms (" << ctx().info().name << ")" << std::endl;

	const auto mismatch = std::mismatch(in.begin(), in.end(), out.begin());
	EXPECT_TRUE(mismatch.first == in.end()) << "first mismatch at word " << (mismatch.first - in.begin());
}
