// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Log messages of the Vulkan backend runtime (no dependency on icl_core).
 *
 * Messages carry a level, a source ("Class::function") and a text. By default messages up to the level Info go to
 * stderr as `[gvl] Warning VoxelList::merge_entries: text`. A program can change the level with set_log_level() and
 * take over the messages (to its own logger, to a test) with set_log_handler(). Handler and level may be changed
 * at any time from any thread; the handler is called without a lock held, from the thread that logs.
 */
#ifndef GPU_VOXELS_VULKAN_LOG_H_INCLUDED
#define GPU_VOXELS_VULKAN_LOG_H_INCLUDED

#include <functional>
#include <sstream>
#include <string>
#include <string_view>

namespace gpu_voxels::vulkan
{
	/// Severity of a message; a message is delivered if its level is at most the configured level (Error < Warning < Info).
	enum class LogLevel
	{
		Error = 0,
		Warning = 1,
		Info = 2
	};

	/// Name of a level ("Error", "Warning", "Info").
	const char* to_string(LogLevel level);

	/// Receives the messages: level, source ("Class::function") and text without a line break.
	using LogHandler = std::function<void(LogLevel level, std::string_view source, std::string_view message)>;

	/// Highest level that is delivered; the default is Info.
	void set_log_level(LogLevel level);
	LogLevel log_level();

	/// True if a message of this level would be delivered now.
	bool log_enabled(LogLevel level);

	/// Replaces the default output (stderr). An empty handler restores the default output.
	void set_log_handler(LogHandler handler);

	/// Delivers a message to the handler (or stderr); does nothing if the level is not enabled. Prefer the GVL_LOG_* macros.
	void log_message(LogLevel level, std::string_view source, std::string_view message);
}

/// GVL_LOG_ERROR(Class, "text " << value): the text is only built if the level is enabled. The source is "Class::<current function>".
#define GVL_LOG(level, class_name, stream_expression) \
	do \
	{ \
		if (::gpu_voxels::vulkan::log_enabled(level)) \
		{ \
			std::ostringstream gvl_log_stream_; \
			gvl_log_stream_ << stream_expression; \
			::gpu_voxels::vulkan::log_message(level, std::string(#class_name "::") + __func__, gvl_log_stream_.str()); \
		} \
	} while (false)

#define GVL_LOG_ERROR(class_name, stream_expression) GVL_LOG(::gpu_voxels::vulkan::LogLevel::Error, class_name, stream_expression)
#define GVL_LOG_WARNING(class_name, stream_expression) GVL_LOG(::gpu_voxels::vulkan::LogLevel::Warning, class_name, stream_expression)
#define GVL_LOG_INFO(class_name, stream_expression) GVL_LOG(::gpu_voxels::vulkan::LogLevel::Info, class_name, stream_expression)

#endif
