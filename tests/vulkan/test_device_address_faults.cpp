// EXPERIMENTS (disabled by default): what does this driver do when a shader dereferences a bad device address?
// A fault may lose the device, so every case must run in its own process:
//   test_vulkan --gtest_also_run_disabled_tests --gtest_filter=DeviceAddressFault.<Case>
// The outcome is printed, not asserted (it is driver behaviour, not a contract); results are recorded in
// Agent/knowledge_base/vulkan_migration/octree_porting_hazards.md.
#include "vulkan_fixture.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>

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
