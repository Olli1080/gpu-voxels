// Shared fixture of the Vulkan backend tests: one Context per test process, validation errors fail the test.
#ifndef GVL_TEST_VULKAN_FIXTURE_H_INCLUDED
#define GVL_TEST_VULKAN_FIXTURE_H_INCLUDED

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <gpu_voxels/vulkan/Context.h>

namespace gvl_test
{
	struct SharedVulkan
	{
		std::unique_ptr<gpu_voxels::vulkan::Context> context;
		bool no_device = false; ///< tests are skipped
		std::string error;      ///< why there is no context
	};

	SharedVulkan& shared_vulkan();

	class VulkanTest : public ::testing::Test
	{
	protected:
		void SetUp() override;
		void TearDown() override;
		[[nodiscard]] gpu_voxels::vulkan::Context& ctx() { return *shared_vulkan().context; }
	};
}

#endif
