#include "vulkan_fixture.h"

#include <cstdlib>
#include <iostream>

using gvl_test::VulkanTest;

TEST_F(VulkanTest, DeviceInfoMeetsPolicy)
{
	const auto& info = ctx().info();
	const auto& limits = ctx().limits();
	std::cout << "[vulkan] subgroup " << info.min_subgroup_size << ".." << info.max_subgroup_size
		<< " (default " << info.subgroup_size << "), maxStorageBufferRange " << limits.maxStorageBufferRange
		<< ", maxWorkGroupInvocations " << limits.maxComputeWorkGroupInvocations
		<< ", device-local heap " << (info.device_local_heap_bytes >> 20) << " MiB"
		<< ", int64 buffer atomics " << (info.shader_int64_buffer_atomics ? "yes" : "no") << std::endl;
	RecordProperty("device", info.name);

	EXPECT_FALSE(info.name.empty());
	EXPECT_GE(info.api_version, static_cast<uint32_t>(VK_API_VERSION_1_3));
	// Vulkan guarantees these minimums.
	EXPECT_GE(limits.maxStorageBufferRange, 128u * 1024 * 1024);
	EXPECT_GE(limits.maxComputeWorkGroupInvocations, 128u);
	EXPECT_GE(limits.maxPushConstantsSize, 128u);
	EXPECT_GT(info.device_local_heap_bytes, 0u);
	if (!std::getenv("GVL_ALLOW_NO_VALIDATION"))
		EXPECT_TRUE(info.validation_enabled);
}

// The oracle of all other tests: prove that a real validation error is detected and counted.
TEST_F(VulkanTest, ValidationLayerDetectsInvalidUsage)
{
	if (!ctx().info().validation_enabled)
		GTEST_SKIP() << "validation disabled";

	const vk::BufferCreateInfo invalid{}; // size 0 and no usage flags violate the spec
	try
	{
		auto buffer = ctx().device().createBufferUnique(invalid);
	}
	catch (const vk::SystemError&)
	{
	}
	EXPECT_GE(ctx().validation_error_count(), 1u);
	EXPECT_FALSE(ctx().validation_messages().empty());
	ctx().reset_validation_state(); // expected error: do not fail this test in TearDown
}
