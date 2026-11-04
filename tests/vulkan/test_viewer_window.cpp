// Window of the in-process viewer (ViewerWindow.h): a real GLFW window with a swapchain on the graphics queue. The window is created hidden. The tests need a
// window system and a Vulkan device that can present; without them they are skipped. Validation errors fail the tests (the first version shared one depth
// image between two frames in flight, which the validation layer reported as a write-after-write hazard).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
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

TEST_F(ViewerWindowTest, KeysControlTheWindowAndTheRestGoToTheHandler)
{
	// GLFW key codes: ESC 256, R 82, S 83, '1' 49
	ViewerWindow window(ctx(), hidden());
	std::vector<int> forwarded;
	window.set_key_handler([&](int key) { forwarded.push_back(key); });
	window.camera().yaw_degrees = 10.0f;
	window.set_home_camera();
	window.camera().yaw_degrees = 77.0f;
	ASSERT_TRUE(window.render_options().shaded);
	window.handle_key(83);
	EXPECT_FALSE(window.render_options().shaded);
	window.handle_key(83);
	EXPECT_TRUE(window.render_options().shaded);
	window.handle_key(82);
	EXPECT_EQ(window.camera().yaw_degrees, 10.0f) << "R returns to the home camera";
	EXPECT_TRUE(forwarded.empty()) << "the keys of the window are not forwarded";
	window.handle_key(49);
	window.handle_key(65);
	EXPECT_EQ(forwarded, (std::vector<int>{ 49, 65 }));
	EXPECT_FALSE(window.should_close());
	window.handle_key(256);
	EXPECT_TRUE(window.should_close());
	// without a handler other keys do nothing
	ViewerWindow plain(ctx(), hidden());
	EXPECT_NO_THROW(plain.handle_key(49));
}

namespace
{
	bool same_camera(const OrbitCamera& a, const OrbitCamera& b)
	{
		return a.target == b.target && a.distance == b.distance && a.yaw_degrees == b.yaw_degrees && a.pitch_degrees == b.pitch_degrees;
	}
}

TEST_F(ViewerWindowTest, TheKeyboardMovesTheCamera)
{
	// GLFW key codes: RIGHT 262, LEFT 263, DOWN 264, UP 265, PAGE_UP 266, PAGE_DOWN 267, HOME 268, MINUS 45, EQUAL 61, KP_SUBTRACT 333, KP_ADD 334, S 83, ESC 256
	ViewerWindow window(ctx(), hidden(320, 240));
	std::vector<int> forwarded;
	window.set_key_handler([&](int key) { forwarded.push_back(key); });
	window.camera().target = { 1.0f, 2.0f, 3.0f };
	window.camera().distance = 8.0f;
	window.set_home_camera();
	OrbitCamera expected = window.camera();
	const float height = static_cast<float>(window.extent().height);
	ASSERT_GT(height, 0.0f);

	window.handle_key(263);
	expected.rotate(-15.0f, 0.0f);
	EXPECT_TRUE(same_camera(window.camera(), expected)) << "left";
	window.handle_key(262);
	expected.rotate(15.0f, 0.0f);
	window.handle_key(265);
	expected.rotate(0.0f, -15.0f);
	window.handle_key(264, 0, true);   // held down
	expected.rotate(0.0f, 15.0f);
	EXPECT_TRUE(same_camera(window.camera(), expected)) << "right, up, down";
	// shift pans
	window.handle_key(263, 1);
	expected.pan(-20.0f, 0.0f, height);
	window.handle_key(265, 1);
	expected.pan(0.0f, -20.0f, height);
	EXPECT_TRUE(same_camera(window.camera(), expected)) << "shift + arrows";
	EXPECT_NE(window.camera().target, (std::array<float, 3>{ 1.0f, 2.0f, 3.0f }));
	// zoom: Page up / + / keypad + closer, Page down / - / keypad - farther
	for (const int key : { 266, 61, 334 })
	{
		window.handle_key(key);
		expected.zoom(1.0f);
		EXPECT_TRUE(same_camera(window.camera(), expected)) << key;
	}
	for (const int key : { 267, 45, 333 })
	{
		window.handle_key(key, 0, true);
		expected.zoom(-1.0f);
		EXPECT_TRUE(same_camera(window.camera(), expected)) << key;
	}
	EXPECT_TRUE(forwarded.empty()) << "camera keys are not forwarded";
	// a held key does not toggle, close or reset anything
	window.handle_key(83, 0, true);
	EXPECT_TRUE(window.render_options().shaded);
	window.handle_key(256, 0, true);
	EXPECT_FALSE(window.should_close());
	window.handle_key(268, 0, true);
	EXPECT_FALSE(same_camera(window.camera(), OrbitCamera{ window.camera().target, 8.0f, window.camera().yaw_degrees, window.camera().pitch_degrees })) << "still moved";
	// Home returns to the home camera, like R
	window.handle_key(268);
	EXPECT_EQ(window.camera().distance, 8.0f);
	EXPECT_EQ(window.camera().target, (std::array<float, 3>{ 1.0f, 2.0f, 3.0f }));
	EXPECT_EQ(window.camera().yaw_degrees, OrbitCamera{}.yaw_degrees);
	EXPECT_EQ(window.camera().pitch_degrees, OrbitCamera{}.pitch_degrees);
	// other keys still reach the handler
	window.handle_key(49);
	EXPECT_EQ(forwarded, (std::vector<int>{ 49 }));

	EXPECT_EQ(window.title(), ViewerWindowOptions{}.title);
	window.set_title("layers: 1 map");
	EXPECT_EQ(window.title(), "layers: 1 map");
}

namespace
{
	std::vector<uint8_t> read_file(const std::filesystem::path& file)
	{
		std::ifstream in(file, std::ios::binary);
		return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	}
}

TEST_F(ViewerWindowTest, AScreenshotShowsWhatTheWindowShows)
{
	BitVoxelMap map(ctx(), { 16, 16, 16 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 1, 1 }, { 8, 8, 8 }, { 14, 3, 9 } }, 5);
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 1000);
	extractor.extract(map, BitPalette::defaults(), cubes);
	const CubeBuffer* buffers[] = { &cubes };

	ViewerWindow window(ctx(), hidden(200, 150));
	window.camera().frame({ 0, 0, 0 }, { 16, 16, 16 });
	const std::filesystem::path file = std::filesystem::temp_directory_path() / "gvl_window_screenshot_test.png";
	std::filesystem::remove(file);

	VoxelRenderer renderer(ctx(), OffscreenTarget::kColourFormat);
	const auto expected_file = [&]
	{
		OffscreenTarget target(ctx(), window.extent().width, window.extent().height);
		target.render(renderer, window.camera().camera(), buffers, window.render_options());
		return encode_png(window.extent().width, window.extent().height, target.read_rgba());
	};

	ASSERT_TRUE(window.draw(buffers));
	EXPECT_FALSE(std::filesystem::exists(file)) << "nothing without a request";
	window.request_screenshot(file);
	ASSERT_TRUE(window.draw(buffers));
	EXPECT_EQ(window.screenshots_written(), 1u);
	EXPECT_TRUE(window.screenshot_error().empty());
	EXPECT_EQ(read_file(file), expected_file()) << "the file is the rendered scene of the window camera";
	std::filesystem::remove(file);
	ASSERT_TRUE(window.draw(buffers));
	EXPECT_FALSE(std::filesystem::exists(file)) << "one request, one file";

	// the options and the camera of the moment are used
	window.handle_key(83);   // S: no shading
	window.camera().yaw_degrees += 40.0f;
	window.request_screenshot(file);
	ASSERT_TRUE(window.draw(buffers));
	EXPECT_EQ(window.screenshots_written(), 2u);
	EXPECT_EQ(read_file(file), expected_file());
	std::filesystem::remove(file);

	// a file that cannot be written does not stop the loop
	window.request_screenshot(std::filesystem::temp_directory_path() / "gvl_no_such_directory_for_screenshots" / "x.png");
	EXPECT_TRUE(window.draw(buffers));
	EXPECT_EQ(window.screenshots_written(), 2u);
	EXPECT_FALSE(window.screenshot_error().empty());
	window.request_screenshot(file);
	EXPECT_TRUE(window.draw(buffers));
	EXPECT_EQ(window.screenshots_written(), 3u);
	EXPECT_TRUE(window.screenshot_error().empty()) << "a later success clears the message";
	std::filesystem::remove(file);

	// key P: gvl_screenshot_<n>.png in the working directory, n counts the files written
	const std::filesystem::path by_key = std::filesystem::current_path() / "gvl_screenshot_4.png";
	std::filesystem::remove(by_key);
	window.handle_key(80, 0, true);   // held down: no request
	ASSERT_TRUE(window.draw(buffers));
	EXPECT_FALSE(std::filesystem::exists(by_key));
	window.handle_key(80);
	ASSERT_TRUE(window.draw(buffers));
	EXPECT_TRUE(std::filesystem::exists(by_key));
	EXPECT_EQ(window.screenshots_written(), 4u);
	std::filesystem::remove(by_key);
}
