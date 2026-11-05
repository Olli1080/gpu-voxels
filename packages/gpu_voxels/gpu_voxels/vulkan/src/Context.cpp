// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/Context.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>

#include <vk_mem_alloc.h>

#include <gpu_voxels/vulkan/Log.h>

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
		constexpr std::size_t kMaxStoredMessages = 64;

		bool has_extension(const std::vector<vk::ExtensionProperties>& list, const char* name)
		{
			return std::any_of(list.begin(), list.end(),
				[&](const vk::ExtensionProperties& p) { return std::strcmp(p.extensionName.data(), name) == 0; });
		}

		const char* device_type_name(vk::PhysicalDeviceType t)
		{
			switch (t)
			{
			case vk::PhysicalDeviceType::eDiscreteGpu: return "discrete";
			case vk::PhysicalDeviceType::eIntegratedGpu: return "integrated";
			case vk::PhysicalDeviceType::eVirtualGpu: return "virtual";
			case vk::PhysicalDeviceType::eCpu: return "cpu";
			default: return "other";
			}
		}

		int device_type_score(vk::PhysicalDeviceType t)
		{
			switch (t)
			{
			case vk::PhysicalDeviceType::eDiscreteGpu: return 3;
			case vk::PhysicalDeviceType::eIntegratedGpu: return 2;
			case vk::PhysicalDeviceType::eVirtualGpu: return 1;
			default: return 0;
			}
		}

		/// What the queue family of the Context must offer.
		struct QueueRequest
		{
			vk::Instance instance;
			bool graphics = false;
			const std::function<bool(vk::Instance, vk::PhysicalDevice, uint32_t)>* filter = nullptr;
		};

		/// Index of the compute queue family (a family without graphics is preferred unless graphics is requested, then a family with graphics and
		/// compute that passes the filter), or -1.
		int find_compute_queue_family(vk::PhysicalDevice pd, const QueueRequest& request)
		{
			const auto families = pd.getQueueFamilyProperties();
			int best = -1;
			for (uint32_t i = 0; i < families.size(); ++i)
			{
				if (!(families[i].queueFlags & vk::QueueFlagBits::eCompute))
					continue;
				if (request.graphics)
				{
					if (!(families[i].queueFlags & vk::QueueFlagBits::eGraphics))
						continue;
					if (request.filter && *request.filter && !(*request.filter)(request.instance, pd, i))
						continue;
					return static_cast<int>(i);
				}
				if (!(families[i].queueFlags & vk::QueueFlagBits::eGraphics))
					return static_cast<int>(i);
				if (best < 0)
					best = static_cast<int>(i);
			}
			return best;
		}

		bool has_device_extensions(vk::PhysicalDevice pd, const std::vector<std::string>& names, std::string& missing)
		{
			if (names.empty())
				return true;
			const auto available = pd.enumerateDeviceExtensionProperties();
			for (const std::string& name : names)
				if (!has_extension(available, name.c_str()))
				{
					missing = name;
					return false;
				}
			return true;
		}

		bool device_meets_requirements(vk::PhysicalDevice pd, const QueueRequest& request, const std::vector<std::string>& device_extensions, std::string& why_not)
		{
			const auto props = pd.getProperties();
			if (props.apiVersion < VK_API_VERSION_1_3)
			{
				why_not = "Vulkan API version < 1.3";
				return false;
			}
			if (find_compute_queue_family(pd, request) < 0)
			{
				why_not = request.graphics ? "no queue family with graphics and compute (and the presentation filter)" : "no compute queue";
				return false;
			}
			if (std::string missing; !has_device_extensions(pd, device_extensions, missing))
			{
				why_not = "device extension " + missing + " not supported";
				return false;
			}
			const auto chain = pd.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
			if (!chain.get<vk::PhysicalDeviceFeatures2>().features.shaderInt64)
			{
				why_not = "shaderInt64 not supported";
				return false;
			}
			if (!chain.get<vk::PhysicalDeviceVulkan13Features>().synchronization2)
			{
				why_not = "synchronization2 not supported";
				return false;
			}
			return true;
		}

		vk::PhysicalDevice select_physical_device(vk::Instance instance, int requested_index, const QueueRequest& request, const std::vector<std::string>& device_extensions)
		{
			const auto devices = instance.enumeratePhysicalDevices();
			if (devices.empty())
				throw NoDeviceError("No Vulkan physical device found");

			if (const char* env = std::getenv("GVL_VULKAN_DEVICE"))
				requested_index = std::atoi(env);

			if (requested_index >= 0)
			{
				if (static_cast<std::size_t>(requested_index) >= devices.size())
					throw NoDeviceError("Vulkan device index " + std::to_string(requested_index) + " out of range");
				std::string why;
				if (!device_meets_requirements(devices[requested_index], request, device_extensions, why))
					throw NoDeviceError("Requested Vulkan device does not meet the requirements: " + why);
				return devices[requested_index];
			}

			vk::PhysicalDevice best;
			int best_score = -1;
			std::string reasons;
			for (const auto& d : devices)
			{
				std::string why;
				const auto props = d.getProperties();
				if (!device_meets_requirements(d, request, device_extensions, why))
				{
					reasons += std::string(props.deviceName.data()) + ": " + why + "; ";
					continue;
				}
				if (const int score = device_type_score(props.deviceType); score > best_score)
				{
					best_score = score;
					best = d;
				}
			}
			if (!best)
				throw NoDeviceError("No Vulkan device meets the requirements (1.3, compute, shaderInt64, synchronization2): " + reasons);
			return best;
		}
	}

	// ---------------------------------------------------------------------------------------------
	struct ValidationState
	{
		std::atomic<uint32_t> errors{ 0 };
		std::atomic<uint32_t> warnings{ 0 };
		mutable std::mutex mutex;
		std::vector<std::string> messages;
		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
		PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger = nullptr;

		~ValidationState()
		{
			if (messenger != VK_NULL_HANDLE && destroy_messenger)
				destroy_messenger(instance, messenger, nullptr);
		}
	};

	namespace
	{
		VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
			VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void* user)
		{
			auto* state = static_cast<ValidationState*>(user);
			const bool is_error = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
			if (is_error)
				++state->errors;
			else
				++state->warnings;

			const std::string text = std::string(is_error ? "[error] " : "[warning] ") + (data && data->pMessage ? data->pMessage : "");
			{
				std::lock_guard lock(state->mutex);
				if (state->messages.size() < kMaxStoredMessages)
					state->messages.push_back(text);
			}
			if (is_error)
				GVL_LOG_ERROR(Context, "Vulkan validation: " << text);
			else
				GVL_LOG_WARNING(Context, "Vulkan validation: " << text);
			return VK_FALSE;
		}
	}

	// ---------------------------------------------------------------------------------------------
	Context::Context(const ContextCreateInfo& create_info)
	{
		// ---- instance ----
		const auto instance_version = vk::enumerateInstanceVersion();
		if (instance_version < VK_API_VERSION_1_3)
			throw NoDeviceError("The Vulkan loader supports less than Vulkan 1.3");

		std::vector<const char*> layers;
		std::vector<const char*> extensions;
		bool validation = false;
		bool validation_features = false;
		bool layer_settings = false;

		if (create_info.enable_validation)
		{
			const auto available_layers = vk::enumerateInstanceLayerProperties();
			const bool layer_present = std::any_of(available_layers.begin(), available_layers.end(),
				[](const vk::LayerProperties& l) { return std::strcmp(l.layerName.data(), kValidationLayer) == 0; });
			if (layer_present)
			{
				const auto base_exts = vk::enumerateInstanceExtensionProperties();
				const auto layer_exts = vk::enumerateInstanceExtensionProperties(std::string(kValidationLayer));
				if (has_extension(base_exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
				{
					validation = true;
					layers.push_back(kValidationLayer);
					extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
					// Synchronization validation: newer layers take it as a layer setting, older ones as a validation feature.
					if (has_extension(base_exts, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME))
					{
						layer_settings = true;
						extensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
					}
					else if (has_extension(layer_exts, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME))
					{
						validation_features = true;
						extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
					}
				}
			}
			if (!validation && create_info.require_validation)
				throw ValidationUnavailableError(std::string(kValidationLayer) +
					" is not available. Install the Vulkan SDK (https://vulkan.lunarg.com) or set GVL_ALLOW_NO_VALIDATION=1.");
			if (!validation)
				GVL_LOG_WARNING(Context, "Vulkan validation layer not available; running unvalidated.");
		}

		for (const std::string& name : create_info.instance_extensions)
			extensions.push_back(name.c_str());

		vk::ApplicationInfo app_info{};
		app_info.setPApplicationName(create_info.application_name.c_str())
			.setPEngineName("gpu_voxels")
			.setApiVersion(VK_API_VERSION_1_3);

		const std::array<vk::ValidationFeatureEnableEXT, 1> enabled_features{ vk::ValidationFeatureEnableEXT::eSynchronizationValidation };
		vk::ValidationFeaturesEXT features_info{};
		features_info.setEnabledValidationFeatures(enabled_features);

		const VkBool32 validate_sync = VK_TRUE;
		vk::LayerSettingEXT sync_setting{};
		sync_setting.setPLayerName(kValidationLayer).setPSettingName("validate_sync")
			.setType(vk::LayerSettingTypeEXT::eBool32);
		sync_setting.valueCount = 1;
		sync_setting.pValues = &validate_sync;
		vk::LayerSettingsCreateInfoEXT settings_info{};
		settings_info.setSettings(sync_setting);

		vk::InstanceCreateInfo instance_info{};
		instance_info.setPApplicationInfo(&app_info)
			.setPEnabledLayerNames(layers)
			.setPEnabledExtensionNames(extensions);
		if (layer_settings)
			instance_info.setPNext(&settings_info);
		else if (validation_features)
			instance_info.setPNext(&features_info);
		instance_ = vk::createInstanceUnique(instance_info);

		// ---- validation messenger ----
		if (validation)
		{
			validation_ = std::make_unique<ValidationState>();
			validation_->instance = *instance_;
			const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
				vkGetInstanceProcAddr(*instance_, "vkCreateDebugUtilsMessengerEXT"));
			validation_->destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
				vkGetInstanceProcAddr(*instance_, "vkDestroyDebugUtilsMessengerEXT"));
			if (!create || !validation_->destroy_messenger)
				throw VulkanError("VK_EXT_debug_utils entry points not found");

			VkDebugUtilsMessengerCreateInfoEXT mi{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
			mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			mi.pfnUserCallback = debug_callback;
			mi.pUserData = validation_.get();
			if (create(*instance_, &mi, nullptr, &validation_->messenger) != VK_SUCCESS)
				throw VulkanError("vkCreateDebugUtilsMessengerEXT failed");
		}

		// ---- physical device, logical device, queue ----
		QueueRequest queue_request;
		queue_request.instance = *instance_;
		queue_request.graphics = create_info.enable_graphics;
		queue_request.filter = &create_info.queue_family_filter;
		physical_device_ = select_physical_device(*instance_, create_info.device_index, queue_request, create_info.device_extensions);
		const auto props2 = physical_device_.getProperties2<vk::PhysicalDeviceProperties2,
			vk::PhysicalDeviceVulkan11Properties, vk::PhysicalDeviceSubgroupSizeControlProperties>();
		const auto& props = props2.get<vk::PhysicalDeviceProperties2>().properties;
		const auto& props11 = props2.get<vk::PhysicalDeviceVulkan11Properties>();
		const auto& props_sg = props2.get<vk::PhysicalDeviceSubgroupSizeControlProperties>();
		limits_ = props.limits;

		info_.name = props.deviceName.data();
		info_.type = device_type_name(props.deviceType);
		info_.vendor_id = props.vendorID;
		info_.device_id = props.deviceID;
		info_.api_version = props.apiVersion;
		info_.driver_version = props.driverVersion;
		info_.queue_family = static_cast<uint32_t>(find_compute_queue_family(physical_device_, queue_request));
		info_.graphics = create_info.enable_graphics;
		info_.subgroup_size = props11.subgroupSize;
		info_.min_subgroup_size = props_sg.minSubgroupSize;
		info_.max_subgroup_size = props_sg.maxSubgroupSize;
		info_.validation_enabled = validation;
		{
			const auto mem = physical_device_.getMemoryProperties();
			for (uint32_t i = 0; i < mem.memoryHeapCount; ++i)
				if (mem.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
					info_.device_local_heap_bytes = std::max<uint64_t>(info_.device_local_heap_bytes, mem.memoryHeaps[i].size);
			const auto f12 = physical_device_.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
			info_.shader_int64_buffer_atomics = f12.get<vk::PhysicalDeviceVulkan12Features>().shaderBufferInt64Atomics;
			info_.buffer_device_address = create_info.enable_buffer_device_address && f12.get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress;
		}

		vk::PhysicalDeviceVulkan13Features features13{};
		features13.setSynchronization2(true).setDynamicRendering(create_info.enable_graphics).setShaderDemoteToHelperInvocation(create_info.enable_graphics); // discard of the sphere shader (SPIR-V demote; mandatory in 1.3) // dynamic rendering is core 1.3: no render pass objects in the viewer
		vk::PhysicalDeviceVulkan12Features features12{};
		features12.setBufferDeviceAddress(info_.buffer_device_address); // optional: only enabled when supported
		features13.setPNext(&features12);
		vk::PhysicalDeviceFeatures2 features2{};
		features2.features.setShaderInt64(true);
		features2.setPNext(&features13);

		const float priority = 1.0f;
		vk::DeviceQueueCreateInfo queue_info{};
		queue_info.setQueueFamilyIndex(info_.queue_family).setQueueCount(1).setPQueuePriorities(&priority);
		std::vector<const char*> device_extension_names;
		for (const std::string& name : create_info.device_extensions)
			device_extension_names.push_back(name.c_str());
		vk::DeviceCreateInfo device_info{};
		device_info.setQueueCreateInfos(queue_info).setPEnabledExtensionNames(device_extension_names).setPNext(&features2);
		device_ = physical_device_.createDeviceUnique(device_info);
		queue_ = device_->getQueue(info_.queue_family, 0);

		// ---- allocator ----
		VmaAllocatorCreateInfo allocator_info{};
		allocator_info.physicalDevice = physical_device_;
		allocator_info.device = *device_;
		allocator_info.instance = *instance_;
		allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
		if (info_.buffer_device_address)
			allocator_info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
		VmaAllocator allocator = nullptr;
		if (vmaCreateAllocator(&allocator_info, &allocator) != VK_SUCCESS)
			throw VulkanError("vmaCreateAllocator failed");
		allocator_ = allocator;

		GVL_LOG_INFO(Context, "Vulkan device '" << info_.name << "' (" << info_.type << "), API "
			<< VK_API_VERSION_MAJOR(info_.api_version) << "." << VK_API_VERSION_MINOR(info_.api_version)
			<< ", validation " << (validation ? "on" : "off"));
	}

	Context::~Context()
	{
		if (device_)
			device_->waitIdle();
		if (allocator_)
			vmaDestroyAllocator(allocator_);
		device_.reset();
		validation_.reset(); // destroys the debug messenger while the instance is still alive
		instance_.reset();
	}

	void Context::execute(const std::function<void(vk::CommandBuffer)>& record)
	{
		vk::CommandPoolCreateInfo pool_info{};
		pool_info.setQueueFamilyIndex(info_.queue_family).setFlags(vk::CommandPoolCreateFlagBits::eTransient);
		const auto pool = device_->createCommandPoolUnique(pool_info);

		vk::CommandBufferAllocateInfo alloc_info{};
		alloc_info.setCommandPool(*pool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1);
		const vk::CommandBuffer cb = device_->allocateCommandBuffers(alloc_info).front();

		vk::CommandBufferBeginInfo begin_info{};
		begin_info.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		cb.begin(begin_info);
		record(cb);
		cb.end();
		submit_and_wait(cb);
	}

	void Context::submit(const vk::SubmitInfo2& info, vk::Fence fence)
	{
		std::lock_guard lock(queue_mutex_);
		queue_.submit2(info, fence);
	}

	vk::Result Context::present(const vk::PresentInfoKHR& info)
	{
		std::lock_guard lock(queue_mutex_);
		return static_cast<vk::Result>(vkQueuePresentKHR(static_cast<VkQueue>(queue_), reinterpret_cast<const VkPresentInfoKHR*>(&info)));
	}

	void Context::submit_and_wait(vk::CommandBuffer command_buffer)
	{
		const auto fence = device_->createFenceUnique({});
		vk::SubmitInfo submit{};
		submit.setCommandBuffers(command_buffer);
		{
			std::lock_guard lock(queue_mutex_);
			queue_.submit(submit, *fence);
		}
		if (device_->waitForFences(*fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
			throw VulkanError("waitForFences did not succeed");
	}

	void Context::wait_idle()
	{
		std::lock_guard lock(queue_mutex_);
		device_->waitIdle();
	}

	uint32_t Context::validation_error_count() const { return validation_ ? validation_->errors.load() : 0; }
	uint32_t Context::validation_warning_count() const { return validation_ ? validation_->warnings.load() : 0; }

	std::vector<std::string> Context::validation_messages() const
	{
		if (!validation_)
			return {};
		std::lock_guard lock(validation_->mutex);
		return validation_->messages;
	}

	void Context::reset_validation_state()
	{
		if (!validation_)
			return;
		std::lock_guard lock(validation_->mutex);
		validation_->errors = 0;
		validation_->warnings = 0;
		validation_->messages.clear();
	}
}
