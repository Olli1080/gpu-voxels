#include <gpu_voxels/vulkan/Log.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>

namespace gpu_voxels::vulkan
{
	namespace
	{
		std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};

		std::mutex g_handler_mutex;
		std::shared_ptr<const LogHandler> g_handler; // null: default output

		std::mutex g_stderr_mutex; // keeps lines of different threads apart
	}

	const char* to_string(LogLevel level)
	{
		switch (level)
		{
		case LogLevel::Error: return "Error";
		case LogLevel::Warning: return "Warning";
		case LogLevel::Info: return "Info";
		}
		return "?";
	}

	void set_log_level(LogLevel level)
	{
		g_level.store(static_cast<int>(level));
	}

	LogLevel log_level()
	{
		return static_cast<LogLevel>(g_level.load());
	}

	bool log_enabled(LogLevel level)
	{
		return static_cast<int>(level) <= g_level.load();
	}

	void set_log_handler(LogHandler handler)
	{
		std::shared_ptr<const LogHandler> next;
		if (handler)
			next = std::make_shared<const LogHandler>(std::move(handler));
		std::lock_guard lock(g_handler_mutex);
		g_handler = std::move(next);
	}

	void log_message(LogLevel level, std::string_view source, std::string_view message)
	{
		if (!log_enabled(level))
			return;

		std::shared_ptr<const LogHandler> handler;
		{
			std::lock_guard lock(g_handler_mutex);
			handler = g_handler;
		}
		if (handler)
		{
			(*handler)(level, source, message);
			return;
		}

		std::lock_guard lock(g_stderr_mutex);
		std::fprintf(stderr, "[gvl] %s %.*s: %.*s\n", to_string(level), static_cast<int>(source.size()), source.data(),
		             static_cast<int>(message.size()), message.data());
		std::fflush(stderr);
	}
}
