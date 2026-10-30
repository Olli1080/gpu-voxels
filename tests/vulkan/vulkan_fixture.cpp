#include "vulkan_fixture.h"

#include <cstdlib>
#include <iostream>

namespace gvl_test
{
	namespace
	{
		SharedVulkan create()
		{
			using namespace gpu_voxels::vulkan;
			SharedVulkan result;
			try
			{
				ContextCreateInfo info;
				// Unvalidated runs are not trustworthy: fail unless explicitly allowed.
				info.require_validation = std::getenv("GVL_ALLOW_NO_VALIDATION") == nullptr;
				result.context = std::make_unique<Context>(info);
				const auto& i = result.context->info();
				std::cout << "[vulkan] device: " << i.name << " (" << i.type << "), API "
					<< VK_API_VERSION_MAJOR(i.api_version) << "." << VK_API_VERSION_MINOR(i.api_version)
					<< ", validation " << (i.validation_enabled ? "on" : "OFF") << std::endl;
			}
			catch (const NoDeviceError& e)
			{
				result.no_device = true;
				result.error = e.what();
			}
			catch (const std::exception& e)
			{
				result.error = e.what();
			}
			return result;
		}

		SharedVulkan g_shared;

		/// Creates the context before the first test and destroys it before static destruction.
		class VulkanEnvironment : public ::testing::Environment
		{
		public:
			void SetUp() override { g_shared = create(); }
			void TearDown() override { g_shared.context.reset(); }
		};

		[[maybe_unused]] const auto* const g_environment = ::testing::AddGlobalTestEnvironment(new VulkanEnvironment);
	}

	SharedVulkan& shared_vulkan()
	{
		return g_shared;
	}

	void VulkanTest::SetUp()
	{
		auto& s = shared_vulkan();
		if (s.no_device)
			GTEST_SKIP() << "No usable Vulkan device: " << s.error;
		ASSERT_TRUE(s.context != nullptr) << "Vulkan context creation failed: " << s.error;
		ctx().reset_validation_state();
	}

	void VulkanTest::TearDown()
	{
		auto& s = shared_vulkan();
		if (!s.context)
			return;
		std::string messages;
		for (const auto& m : ctx().validation_messages())
			messages += m + "\n";
		EXPECT_EQ(ctx().validation_error_count(), 0u) << "Vulkan validation errors:\n" << messages;
		if (ctx().validation_warning_count() > 0)
			std::cout << "[vulkan] " << ctx().validation_warning_count() << " validation warning(s):\n" << messages;
	}
}
