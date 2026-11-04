// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Window of the in-process Vulkan viewer: GLFW window, swapchain, depth buffer, frame loop and mouse control of an OrbitCamera. The cubes
 *         come from the CubeBuffers of Viewer.h, i.e. straight from the GPU buffers of the maps of the same Context. No separate viewer process.
 *
 * Typical use:
 * \code
 *   ViewerWindow::initialize();                       // GLFW; throws if there is no window system or no Vulkan support
 *   Context context(ViewerWindow::context_requirements());
 *   BitVoxelMap map(context, ...);                    // the maps live on the same Context
 *   CubeExtractor extractor(context);
 *   CubeBuffer cubes(context, 1 << 20);
 *   ViewerWindow window(context);
 *   while (!window.should_close())
 *   {
 *       window.poll_events();
 *       extractor.extract(map, BitPalette::defaults(), cubes);   // whenever the map changed
 *       const CubeBuffer* buffers[] = { &cubes };
 *       window.draw(buffers);
 *   }
 * \endcode
 */
#ifndef GPU_VOXELS_VULKAN_VIEWER_WINDOW_H_INCLUDED
#define GPU_VOXELS_VULKAN_VIEWER_WINDOW_H_INCLUDED

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/OrbitCamera.h>
#include <gpu_voxels/vulkan/Viewer.h>

struct GLFWwindow;

namespace gpu_voxels::vulkan
{
	struct ViewerWindowOptions
	{
		std::string title = "GPU-Voxels viewer";
		uint32_t width = 1280;
		uint32_t height = 720;
		bool vsync = true;     ///< FIFO present mode; false prefers MAILBOX / IMMEDIATE
		bool visible = true;   ///< false: the window is created hidden (tests, screenshots)
	};

	class ViewerWindow
	{
	public:
		/// Initialises GLFW. \throws VulkanError if GLFW cannot be initialised or has no Vulkan support. Safe to call repeatedly.
		static void initialize();
		/// `base` plus what a window needs: the instance extensions of GLFW, VK_KHR_swapchain, a graphics + compute queue family that can present.
		/// Calls initialize().
		[[nodiscard]] static ContextCreateInfo context_requirements(ContextCreateInfo base = {});

		/// \throws VulkanError if the Context does not meet context_requirements(), or no window / surface can be created
		explicit ViewerWindow(Context& context, const ViewerWindowOptions& options = {});
		~ViewerWindow();
		ViewerWindow(const ViewerWindow&) = delete;
		ViewerWindow& operator=(const ViewerWindow&) = delete;

		[[nodiscard]] bool should_close() const;
		void request_close();
		/// Process the window events (mouse and keys move the camera; ESC closes, S toggles the shading, R resets the camera).
		void poll_events();

		[[nodiscard]] OrbitCamera& camera() { return camera_; }
		/// The camera that key R returns to (the camera at the time of the call; at construction the default).
		void set_home_camera() { initial_camera_ = camera_; }
		[[nodiscard]] RenderOptions& render_options() { return render_options_; }

		/// Draw one frame of the cubes and present it. Returns false if no frame was presented (window minimised, swapchain out of date: the swapchain is
		/// recreated and the next call draws).
		bool draw(std::span<const CubeBuffer* const> buffers);

		[[nodiscard]] vk::Extent2D extent() const { return extent_; }
		[[nodiscard]] vk::Format format() const { return format_; }
		[[nodiscard]] uint64_t frames_presented() const { return frames_presented_; }
		[[nodiscard]] uint32_t swapchain_creations() const { return swapchain_creations_; }
		/// Window size in pixels (framebuffer).
		[[nodiscard]] std::array<uint32_t, 2> framebuffer_size() const;
		/// Test hook: change the size of the window; the swapchain follows with the next draw().
		void resize(uint32_t width, uint32_t height);

	private:
		struct Impl;
		void create_swapchain();
		void destroy_swapchain();

		Context& context_;
		ViewerWindowOptions options_;
		GLFWwindow* window_ = nullptr;
		OrbitCamera camera_;
		OrbitCamera initial_camera_;
		OrbitController controller_;
		RenderOptions render_options_;
		vk::Extent2D extent_{};
		vk::Format format_ = vk::Format::eUndefined;
		uint64_t frames_presented_ = 0;
		uint32_t swapchain_creations_ = 0;
		std::unique_ptr<Impl> impl_;
	};
}

#endif
