// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan_viewer/ViewerWindow.h>

#include <algorithm>
#include <array>
#include <mutex>

#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kFramesInFlight = 2;
		constexpr vk::Format kDepthFormat = vk::Format::eD32Sfloat;

		std::mutex g_glfw_mutex;
		bool g_glfw_ready = false;

		void check(VkResult result, const char* what)
		{
			if (result != VK_SUCCESS)
				throw VulkanError(std::string(what) + " failed with VkResult " + std::to_string(static_cast<int>(result)));
		}

		void glfw_error_callback(int code, const char* description)
		{
			// stored for the exception that follows a failing call; printing here would be lost in tests
			(void)code;
			(void)description;
		}
	}

	struct ViewerWindow::Impl
	{
		struct Frame
		{
			vk::UniqueCommandPool pool;
			vk::CommandBuffer command_buffer;
			vk::UniqueSemaphore image_available;
			vk::UniqueFence in_flight;
		};

		vk::SurfaceKHR surface;
		vk::UniqueSwapchainKHR swapchain;
		std::vector<vk::Image> images;
		std::vector<vk::UniqueImageView> views;
		std::vector<vk::UniqueSemaphore> render_finished;
		/// One depth buffer per frame in flight: frames that overlap on the GPU must not share an attachment (the validation layer reports the layout transition of
		/// the second frame as a write-after-write hazard against the first one).
		struct Depth
		{
			vk::Image image;
			VmaAllocation allocation = nullptr;
			vk::UniqueImageView view;
		};
		std::array<Depth, kFramesInFlight> depth;
		std::unique_ptr<VoxelRenderer> renderer;
		std::array<Frame, kFramesInFlight> frames;
		uint32_t frame_index = 0;
		bool needs_recreate = false;
	};

	void ViewerWindow::initialize()
	{
		std::lock_guard lock(g_glfw_mutex);
		if (g_glfw_ready)
			return;
		glfwSetErrorCallback(glfw_error_callback);
		if (glfwInit() != GLFW_TRUE)
			throw VulkanError("ViewerWindow: glfwInit failed (no window system?)");
		if (glfwVulkanSupported() != GLFW_TRUE)
		{
			glfwTerminate();
			throw VulkanError("ViewerWindow: GLFW finds no Vulkan loader / presentation support");
		}
		g_glfw_ready = true;
	}

	ContextCreateInfo ViewerWindow::context_requirements(ContextCreateInfo base)
	{
		initialize();
		uint32_t count = 0;
		const char** required = glfwGetRequiredInstanceExtensions(&count);
		if (!required)
			throw VulkanError("ViewerWindow: GLFW reports no instance extensions for surfaces");
		for (uint32_t i = 0; i < count; ++i)
			if (std::find(base.instance_extensions.begin(), base.instance_extensions.end(), required[i]) == base.instance_extensions.end())
				base.instance_extensions.emplace_back(required[i]);
		if (std::find(base.device_extensions.begin(), base.device_extensions.end(), VK_KHR_SWAPCHAIN_EXTENSION_NAME) == base.device_extensions.end())
			base.device_extensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		base.enable_graphics = true;
		const auto previous = base.queue_family_filter;
		base.queue_family_filter = [previous](vk::Instance instance, vk::PhysicalDevice device, uint32_t family)
		{
			if (previous && !previous(instance, device, family))
				return false;
			return glfwGetPhysicalDevicePresentationSupport(static_cast<VkInstance>(instance), static_cast<VkPhysicalDevice>(device), family) == GLFW_TRUE;
		};
		return base;
	}

	ViewerWindow::ViewerWindow(Context& context, const ViewerWindowOptions& options)
		: context_(context), options_(options), controller_(camera_), impl_(std::make_unique<Impl>())
	{
		initialize();
		if (!context.info().graphics)
			throw VulkanError("ViewerWindow: the Context was not created with ViewerWindow::context_requirements()");
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_VISIBLE, options.visible ? GLFW_TRUE : GLFW_FALSE);
		window_ = glfwCreateWindow(static_cast<int>(options.width), static_cast<int>(options.height), options.title.c_str(), nullptr, nullptr);
		if (!window_)
			throw VulkanError("ViewerWindow: glfwCreateWindow failed");
		glfwSetWindowUserPointer(window_, this);

		try
		{
			VkSurfaceKHR surface = VK_NULL_HANDLE;
			check(glfwCreateWindowSurface(static_cast<VkInstance>(context.instance()), window_, nullptr, &surface), "glfwCreateWindowSurface");
			impl_->surface = surface;
			if (!context.physical_device().getSurfaceSupportKHR(context.queue_family(), impl_->surface))
				throw VulkanError("ViewerWindow: the queue family of the Context cannot present to this window (create the Context with ViewerWindow::context_requirements())");

			const vk::Device device = context.device();
			for (auto& frame : impl_->frames)
			{
				vk::CommandPoolCreateInfo pool_info{};
				pool_info.setQueueFamilyIndex(context.queue_family()).setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
				frame.pool = device.createCommandPoolUnique(pool_info);
				frame.command_buffer = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{ *frame.pool, vk::CommandBufferLevel::ePrimary, 1 }).front();
				frame.image_available = device.createSemaphoreUnique({});
				frame.in_flight = device.createFenceUnique(vk::FenceCreateInfo{ vk::FenceCreateFlagBits::eSignaled });
			}
			create_swapchain();
		}
		catch (...)
		{
			if (impl_->surface)
			{
				context.wait_idle();
				impl_->swapchain.reset();
				context.instance().destroySurfaceKHR(impl_->surface);
			}
			glfwDestroyWindow(window_);
			throw;
		}

		glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) { static_cast<ViewerWindow*>(glfwGetWindowUserPointer(w))->impl_->needs_recreate = true; });
		glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int button, int action, int)
		{
			auto* self = static_cast<ViewerWindow*>(glfwGetWindowUserPointer(w));
			const bool pressed = action == GLFW_PRESS;
			if (button == GLFW_MOUSE_BUTTON_LEFT)
				self->controller_.button(MouseButton::Left, pressed);
			else if (button == GLFW_MOUSE_BUTTON_RIGHT)
				self->controller_.button(MouseButton::Right, pressed);
			else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
				self->controller_.button(MouseButton::Middle, pressed);
		});
		glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double x, double y)
		{
			auto* self = static_cast<ViewerWindow*>(glfwGetWindowUserPointer(w));
			int window_width = 0, window_height = 0;
			glfwGetWindowSize(w, &window_width, &window_height);
			self->controller_.cursor(static_cast<float>(x), static_cast<float>(y), static_cast<float>(window_height));
		});
		glfwSetScrollCallback(window_, [](GLFWwindow* w, double, double dy) { static_cast<ViewerWindow*>(glfwGetWindowUserPointer(w))->controller_.scroll(static_cast<float>(dy)); });
		glfwSetKeyCallback(window_, [](GLFWwindow* w, int key, int, int action, int)
		{
			if (action != GLFW_PRESS)
				return;
			auto* self = static_cast<ViewerWindow*>(glfwGetWindowUserPointer(w));
			if (key == GLFW_KEY_ESCAPE)
				glfwSetWindowShouldClose(w, GLFW_TRUE);
			else if (key == GLFW_KEY_S)
				self->render_options_.shaded = !self->render_options_.shaded;
			else if (key == GLFW_KEY_R)
				self->camera_ = self->initial_camera_;
		});
	}

	ViewerWindow::~ViewerWindow()
	{
		context_.wait_idle();
		destroy_swapchain();
		for (auto& frame : impl_->frames)
		{
			frame.in_flight.reset();
			frame.image_available.reset();
			frame.pool.reset();
		}
		impl_->renderer.reset();
		if (impl_->surface)
			context_.instance().destroySurfaceKHR(impl_->surface);
		if (window_)
			glfwDestroyWindow(window_);
	}

	bool ViewerWindow::should_close() const { return glfwWindowShouldClose(window_) == GLFW_TRUE; }

	void ViewerWindow::request_close() { glfwSetWindowShouldClose(window_, GLFW_TRUE); }

	void ViewerWindow::poll_events() { glfwPollEvents(); }

	std::array<uint32_t, 2> ViewerWindow::framebuffer_size() const
	{
		int w = 0, h = 0;
		glfwGetFramebufferSize(window_, &w, &h);
		return { static_cast<uint32_t>(std::max(w, 0)), static_cast<uint32_t>(std::max(h, 0)) };
	}

	void ViewerWindow::resize(uint32_t width, uint32_t height)
	{
		glfwSetWindowSize(window_, static_cast<int>(width), static_cast<int>(height));
		glfwPollEvents();
		impl_->needs_recreate = true;
	}

	void ViewerWindow::create_swapchain()
	{
		const vk::PhysicalDevice pd = context_.physical_device();
		const vk::Device device = context_.device();
		const auto caps = pd.getSurfaceCapabilitiesKHR(impl_->surface);

		const auto size = framebuffer_size();
		vk::Extent2D extent = caps.currentExtent;
		if (extent.width == UINT32_MAX)
		{
			extent.width = std::clamp(size[0], caps.minImageExtent.width, caps.maxImageExtent.width);
			extent.height = std::clamp(size[1], caps.minImageExtent.height, caps.maxImageExtent.height);
		}
		if (extent.width == 0 || extent.height == 0)
		{
			extent_ = vk::Extent2D{ 0, 0 }; // minimised: nothing to draw into
			impl_->needs_recreate = true;
			return;
		}

		const auto formats = pd.getSurfaceFormatsKHR(impl_->surface);
		if (formats.empty())
			throw VulkanError("ViewerWindow: the surface has no formats");
		vk::SurfaceFormatKHR chosen = formats.front();
		for (const auto& candidate : formats)
			if (candidate.format == vk::Format::eB8G8R8A8Unorm && candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
				chosen = candidate;
		if (chosen.format != vk::Format::eB8G8R8A8Unorm)
			for (const auto& candidate : formats)
				if (candidate.format == vk::Format::eR8G8B8A8Unorm && candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
					chosen = candidate;

		const auto modes = pd.getSurfacePresentModesKHR(impl_->surface);
		const auto has_mode = [&](vk::PresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
		vk::PresentModeKHR mode = vk::PresentModeKHR::eFifo;
		if (!options_.vsync)
			mode = has_mode(vk::PresentModeKHR::eMailbox) ? vk::PresentModeKHR::eMailbox : (has_mode(vk::PresentModeKHR::eImmediate) ? vk::PresentModeKHR::eImmediate : mode);

		uint32_t image_count = caps.minImageCount + 1;
		if (caps.maxImageCount > 0)
			image_count = std::min(image_count, caps.maxImageCount);
		const vk::CompositeAlphaFlagBitsKHR alpha = (caps.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque) ? vk::CompositeAlphaFlagBitsKHR::eOpaque
		                                                                                                                     : vk::CompositeAlphaFlagBitsKHR::eInherit;

		context_.wait_idle();
		vk::SwapchainCreateInfoKHR info{};
		info.setSurface(impl_->surface).setMinImageCount(image_count).setImageFormat(chosen.format).setImageColorSpace(chosen.colorSpace).setImageExtent(extent)
			.setImageArrayLayers(1).setImageUsage(vk::ImageUsageFlagBits::eColorAttachment).setImageSharingMode(vk::SharingMode::eExclusive)
			.setPreTransform(caps.currentTransform).setCompositeAlpha(alpha).setPresentMode(mode).setClipped(VK_TRUE).setOldSwapchain(*impl_->swapchain);
		auto new_swapchain = device.createSwapchainKHRUnique(info);
		impl_->views.clear();
		impl_->render_finished.clear();
		impl_->swapchain = std::move(new_swapchain);
		impl_->images = device.getSwapchainImagesKHR(*impl_->swapchain);
		for (const vk::Image image : impl_->images)
		{
			vk::ImageViewCreateInfo view_info{};
			view_info.setImage(image).setViewType(vk::ImageViewType::e2D).setFormat(chosen.format).setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
			impl_->views.push_back(device.createImageViewUnique(view_info));
			impl_->render_finished.push_back(device.createSemaphoreUnique({}));
		}

		// depth buffers of the new size
		for (auto& depth : impl_->depth)
		{
			depth.view.reset();
			if (depth.allocation)
				vmaDestroyImage(context_.allocator(), static_cast<VkImage>(depth.image), depth.allocation);
			VkImageCreateInfo depth_info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
			depth_info.imageType = VK_IMAGE_TYPE_2D;
			depth_info.format = static_cast<VkFormat>(kDepthFormat);
			depth_info.extent = { extent.width, extent.height, 1 };
			depth_info.mipLevels = 1;
			depth_info.arrayLayers = 1;
			depth_info.samples = VK_SAMPLE_COUNT_1_BIT;
			depth_info.tiling = VK_IMAGE_TILING_OPTIMAL;
			depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			depth_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			depth_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			VmaAllocationCreateInfo alloc{};
			alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
			VkImage raw_depth = VK_NULL_HANDLE;
			depth.allocation = nullptr;
			check(vmaCreateImage(context_.allocator(), &depth_info, &alloc, &raw_depth, &depth.allocation, nullptr), "vmaCreateImage(depth)");
			depth.image = raw_depth;
			vk::ImageViewCreateInfo depth_view_info{};
			depth_view_info.setImage(depth.image).setViewType(vk::ImageViewType::e2D).setFormat(kDepthFormat).setSubresourceRange({ vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 });
			depth.view = device.createImageViewUnique(depth_view_info);
		}

		if (!impl_->renderer || impl_->renderer->colour_format() != chosen.format)
			impl_->renderer = std::make_unique<VoxelRenderer>(context_, chosen.format, kDepthFormat);
		extent_ = extent;
		format_ = chosen.format;
		impl_->needs_recreate = false;
		++swapchain_creations_;
	}

	void ViewerWindow::destroy_swapchain()
	{
		impl_->views.clear();
		impl_->render_finished.clear();
		impl_->images.clear();
		for (auto& depth : impl_->depth)
		{
			depth.view.reset();
			if (depth.allocation)
			{
				vmaDestroyImage(context_.allocator(), static_cast<VkImage>(depth.image), depth.allocation);
				depth.allocation = nullptr;
			}
		}
		impl_->swapchain.reset();
	}

	bool ViewerWindow::draw(std::span<const CubeBuffer* const> buffers)
	{
		const auto size = framebuffer_size();
		if (size[0] == 0 || size[1] == 0)
			return false; // minimised
		if (impl_->needs_recreate || extent_.width != size[0] || extent_.height != size[1])
		{
			create_swapchain();
			if (extent_.width == 0)
				return false;
		}

		const vk::Device device = context_.device();
		auto& frame = impl_->frames[impl_->frame_index];
		if (device.waitForFences(*frame.in_flight, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
			throw VulkanError("ViewerWindow: waiting for the frame fence failed");

		uint32_t image_index = 0;
		try
		{
			const auto acquired = device.acquireNextImageKHR(*impl_->swapchain, UINT64_MAX, *frame.image_available, nullptr);
			image_index = acquired.value;
			if (acquired.result == vk::Result::eSuboptimalKHR)
				impl_->needs_recreate = true;
		}
		catch (const vk::OutOfDateKHRError&)
		{
			impl_->needs_recreate = true;
			return false;
		}
		device.resetFences(*frame.in_flight);

		const vk::CommandBuffer cb = frame.command_buffer;
		cb.reset();
		cb.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
		using Stage = vk::PipelineStageFlagBits2;
		using Access = vk::AccessFlagBits2;
		const auto barrier = [](vk::Image image, vk::ImageAspectFlags aspect, vk::ImageLayout from, vk::ImageLayout to, vk::PipelineStageFlags2 src_stage,
			vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access)
		{
			vk::ImageMemoryBarrier2 b{};
			b.setSrcStageMask(src_stage).setSrcAccessMask(src_access).setDstStageMask(dst_stage).setDstAccessMask(dst_access).setOldLayout(from).setNewLayout(to)
				.setImage(image).setSubresourceRange({ aspect, 0, 1, 0, 1 }).setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
			return b;
		};
		const std::array<vk::ImageMemoryBarrier2, 2> to_attachment{
			barrier(impl_->images[image_index], vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
				Stage::eColorAttachmentOutput, {}, Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite),
			barrier(impl_->depth[impl_->frame_index].image, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
				Stage::eEarlyFragmentTests | Stage::eLateFragmentTests, {}, Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
				Access::eDepthStencilAttachmentWrite | Access::eDepthStencilAttachmentRead) };
		cb.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(to_attachment));

		impl_->renderer->record(cb, *impl_->views[image_index], *impl_->depth[impl_->frame_index].view, extent_, camera_.camera(), buffers, render_options_);

		const auto to_present = barrier(impl_->images[image_index], vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
			Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite, Stage::eNone, {});
		cb.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(to_present));
		cb.end();

		const vk::SemaphoreSubmitInfo wait{ *frame.image_available, 0, Stage::eColorAttachmentOutput };
		const vk::SemaphoreSubmitInfo signal{ *impl_->render_finished[image_index], 0, Stage::eAllCommands };
		const vk::CommandBufferSubmitInfo command_info{ cb };
		vk::SubmitInfo2 submit{};
		submit.setWaitSemaphoreInfos(wait).setCommandBufferInfos(command_info).setSignalSemaphoreInfos(signal);
		context_.submit(submit, *frame.in_flight);

		const vk::SwapchainKHR swapchain = *impl_->swapchain;
		const vk::Semaphore render_finished = *impl_->render_finished[image_index];
		vk::PresentInfoKHR present{};
		present.setWaitSemaphores(render_finished).setSwapchains(swapchain).setImageIndices(image_index);
		const vk::Result result = context_.present(present);
		if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
			impl_->needs_recreate = true;
		else if (result != vk::Result::eSuccess)
			throw VulkanError("ViewerWindow: presenting failed with VkResult " + std::to_string(static_cast<int>(result)));

		impl_->frame_index = (impl_->frame_index + 1) % kFramesInFlight;
		++frames_presented_;
		return true;
	}
}
