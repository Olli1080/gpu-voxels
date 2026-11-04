// Window of the in-process viewer (ViewerWindow.h): a real GLFW window with a swapchain on the graphics queue. The window is created hidden. The tests need a
// window system and a Vulkan device that can present; without them they are skipped. Validation errors fail the tests (the first version shared one depth
// image between two frames in flight, which the validation layer reported as a write-after-write hazard).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/ImageFile.h>
#include <gpu_voxels/vulkan/Viewer.h>
#include <gpu_voxels/vulkan_viewer/ViewerWindow.h>

using namespace gpu_voxels::vulkan;

namespace
{
	struct WindowContext
	{
		std::unique_ptr<Context> context;
		std::string skip_reason;
	};

	WindowContext& window_context()
	{
		static WindowContext result = []
		{
			WindowContext r;
			try
			{
				ContextCreateInfo base;
				base.require_validation = std::getenv("GVL_ALLOW_NO_VALIDATION") == nullptr;
				r.context = std::make_unique<Context>(ViewerWindow::context_requirements(base));
			}
			catch (const std::exception& e)
			{
				r.skip_reason = e.what();
			}
			return r;
		}();
		return result;
	}

	class ViewerWindowTest : public ::testing::Test
	{
	protected:
		void SetUp() override
		{
			if (!window_context().context)
				GTEST_SKIP() << "No window system / presentable Vulkan device: " << window_context().skip_reason;
			ctx().reset_validation_state();
		}

		void TearDown() override
		{
			if (!window_context().context)
				return;
			std::string messages;
			for (const auto& m : ctx().validation_messages())
				messages += m + "\n";
			EXPECT_EQ(ctx().validation_error_count(), 0u) << "Vulkan validation errors:\n" << messages;
		}

		[[nodiscard]] Context& ctx() { return *window_context().context; }
	};

	ViewerWindowOptions hidden(uint32_t w = 320, uint32_t h = 240)
	{
		ViewerWindowOptions options;
		options.width = w;
		options.height = h;
		options.visible = false;
		return options;
	}
}

TEST_F(ViewerWindowTest, DrawsAndPresentsFrames)
{
	BitVoxelMap map(ctx(), { 16, 16, 16 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 1, 1 }, { 8, 8, 8 }, { 14, 3, 9 } }, 5);
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 1000);
	extractor.extract(map, BitPalette::defaults(), cubes);

	ViewerWindow window(ctx(), hidden());
	EXPECT_EQ(window.format(), vk::Format::eB8G8R8A8Unorm);
	window.camera().frame({ 0, 0, 0 }, { 16, 16, 16 });
	const CubeBuffer* buffers[] = { &cubes };
	for (int i = 0; i < 10; ++i)
	{
		window.poll_events();
		EXPECT_TRUE(window.draw(buffers)) << i;
	}
	EXPECT_EQ(window.frames_presented(), 10u);
	EXPECT_EQ(window.swapchain_creations(), 1u);
	EXPECT_FALSE(window.should_close());
	window.request_close();
	EXPECT_TRUE(window.should_close());
}

TEST_F(ViewerWindowTest, TheSwapchainFollowsTheWindowSize)
{
	CubeBuffer cubes(ctx(), 10);
	ViewerWindow window(ctx(), hidden(320, 240));
	const CubeBuffer* buffers[] = { &cubes };
	ASSERT_TRUE(window.draw(buffers));
	const vk::Extent2D first = window.extent();
	EXPECT_EQ(first.width, window.framebuffer_size()[0]);
	EXPECT_EQ(first.height, window.framebuffer_size()[1]);
	const uint32_t creations = window.swapchain_creations();

	window.resize(400, 300);
	ASSERT_TRUE(window.draw(buffers));
	EXPECT_EQ(window.swapchain_creations(), creations + 1);
	EXPECT_EQ(window.extent().width, window.framebuffer_size()[0]);
	EXPECT_EQ(window.extent().height, window.framebuffer_size()[1]);
	EXPECT_NE(window.extent().width, first.width);
	// frames keep working after the recreation
	for (int i = 0; i < 5; ++i)
		EXPECT_TRUE(window.draw(buffers));
}

TEST_F(ViewerWindowTest, SeveralWindowsOnOneContext)
{
	CubeBuffer cubes(ctx(), 10);
	const CubeBuffer* buffers[] = { &cubes };
	ViewerWindow a(ctx(), hidden(200, 100));
	ViewerWindow b(ctx(), hidden(256, 128));
	for (int i = 0; i < 4; ++i)
	{
		EXPECT_TRUE(a.draw(buffers));
		EXPECT_TRUE(b.draw(buffers));
	}
	EXPECT_EQ(a.frames_presented(), 4u);
	EXPECT_EQ(b.frames_presented(), 4u);
}

TEST_F(ViewerWindowTest, ARendererOfAComputeOnlyContextIsRefused)
{
	ContextCreateInfo info;
	info.require_validation = std::getenv("GVL_ALLOW_NO_VALIDATION") == nullptr;
	Context compute_only(info);
	EXPECT_THROW(ViewerWindow(compute_only, hidden()), VulkanError);
}

TEST_F(ViewerWindowTest, TheContextRequirementsAreAddedToTheBaseInfo)
{
	ContextCreateInfo base;
	base.application_name = "viewer test";
	base.instance_extensions.push_back("VK_KHR_surface");
	const ContextCreateInfo info = ViewerWindow::context_requirements(base);
	EXPECT_EQ(info.application_name, "viewer test");
	EXPECT_TRUE(info.enable_graphics);
	EXPECT_NE(std::find(info.device_extensions.begin(), info.device_extensions.end(), std::string("VK_KHR_swapchain")), info.device_extensions.end());
	EXPECT_EQ(std::count(info.instance_extensions.begin(), info.instance_extensions.end(), std::string("VK_KHR_surface")), 1) << "no duplicates";
	EXPECT_TRUE(static_cast<bool>(info.queue_family_filter));
}
