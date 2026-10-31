// EXPERIMENTS (disabled by default): what does this driver do when a shader dereferences a bad device address?
// A fault may lose the device, so every case must run in its own process:
//   test_vulkan --gtest_also_run_disabled_tests --gtest_filter=VulkanTest.DISABLED_<Case>
// The outcome is printed, not asserted (it is driver behaviour, not a contract); results are recorded in
// Agent/knowledge_base/known_issues.md (K25, K28, K29) and Agent/Analysis/2026-10-31_machine_baseline.md.
#include "vulkan_fixture.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;

namespace
{
	struct ReadPush { uint64_t address; };

	void read_word(Context& ctx, uint64_t address, const char* what)
	{
		DeviceBuffer out(ctx, 4);
		out.upload<uint32_t>(std::vector<uint32_t>{ 0x11111111u });
		ComputePipeline pipeline(ctx, ComputePipeline::load_spirv(std::filesystem::path(GVL_TEST_SHADER_DIR) / "bda_read.comp.spv"), 1, sizeof(ReadPush));
		CommandBatch batch(ctx);
		batch.dispatch(pipeline, { &out }, 1, ReadPush{ address });
		try
		{
			batch.submit_and_wait();
			std::cout << "[fault-experiment] " << what << ": completed, value 0x" << std::hex << out.download_vector<uint32_t>(1)[0] << std::dec << std::endl;
		}
		catch (const std::exception& e)
		{
			std::cout << "[fault-experiment] " << what << ": exception: " << e.what() << std::endl;
		}
	}
}

TEST_F(VulkanTest, DISABLED_ReadPastTheEndOfABuffer)
{
	DeviceBuffer small(ctx(), 64);
	small.upload<uint32_t>(std::vector<uint32_t>(16, 0xAAAAAAAAu));
	read_word(ctx(), small.device_address() + 64, "read 64 bytes past a 64 byte buffer");
}

TEST_F(VulkanTest, DISABLED_ReadFarPastTheEndOfABuffer)
{
	DeviceBuffer small(ctx(), 64);
	read_word(ctx(), small.device_address() + (64ull << 20), "read 64 MiB past a 64 byte buffer");
}

TEST_F(VulkanTest, DISABLED_ReadNullAddress)
{
	read_word(ctx(), 0, "read address 0");
}

TEST_F(VulkanTest, DISABLED_ReadAfterBufferWasFreed)
{
	uint64_t address = 0;
	{
		DeviceBuffer gone(ctx(), 4096);
		address = gone.device_address();
	}
	read_word(ctx(), address, "read a freed buffer");
}

// K29: the format of device addresses of this driver (the octree sign-extends bit 47 when it reads a child pointer).
TEST_F(VulkanTest, DISABLED_PrintDeviceAddressFormat)
{
	for (uint64_t bytes : { 64ull, 4096ull, 1ull << 20, 1ull << 28 })
	{
		DeviceBuffer buffer(ctx(), bytes);
		const uint64_t a = buffer.device_address();
		std::cout << "[fault-experiment] address of a " << bytes << " byte buffer: 0x" << std::hex << a << std::dec
			<< ", upper 16 bits 0x" << std::hex << (a >> 48) << std::dec << ", bit 47 " << ((a >> 47) & 1) << std::endl;
	}
}

// K25: one invocation walks a whole ray; longer and longer rays on a 2^24 x 1 x 1 map until the watchdog (if any) strikes. The device
// may be lost, so run it in its own process; the lengths are printed with the time of each insert.
TEST_F(VulkanTest, DISABLED_LongRayAgainstTheWatchdog)
{
	for (uint32_t length : { 1u << 16, 1u << 20, 1u << 22, 1u << 23, (1u << 24) - 1u })
	{
		try
		{
			ProbVoxelMap map(ctx(), { 1u << 24, 1, 1 }, 1.0f);
			const std::vector<Point3f> hit{ { float(length), 0.5f, 0.5f } };
			const auto t0 = std::chrono::steady_clock::now();
			map.insert_sensor_data(hit, Point3f{ 0.5f, 0.5f, 0.5f }, true);
			const auto t1 = std::chrono::steady_clock::now();
			std::cout << "[fault-experiment] ray of " << length << " voxels: completed in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
		}
		catch (const std::exception& e)
		{
			std::cout << "[fault-experiment] ray of " << length << " voxels: exception: " << e.what() << std::endl;
			break;
		}
	}
}
