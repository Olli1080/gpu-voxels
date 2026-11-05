// Log messages of the Vulkan backend (gpu_voxels/vulkan/Log.h): levels, handler, default output, thread safety, and the messages of
// the classes that used the icl_core stream VulkanLog before (same texts, same sources).
#include "vulkan_fixture.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/GpuVoxels.h>
#include <gpu_voxels/vulkan/Log.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;

namespace
{
	/// The level as the program started with it (read during static initialisation, before any test changes it).
	const LogLevel kLevelAtStart = log_level();

	struct Message
	{
		LogLevel level;
		std::string source;
		std::string text;
	};

	/// Installs a handler that records the messages; restores the defaults (empty handler, level Info) when it goes out of scope.
	class LogCapture
	{
	public:
		LogCapture()
		{
			set_log_handler([this](LogLevel level, std::string_view source, std::string_view text) {
				std::lock_guard lock(mutex_);
				messages_.push_back({ level, std::string(source), std::string(text) });
			});
		}
		~LogCapture()
		{
			set_log_handler({});
			set_log_level(LogLevel::Info);
		}
		LogCapture(const LogCapture&) = delete;
		LogCapture& operator=(const LogCapture&) = delete;

		[[nodiscard]] std::vector<Message> messages() const
		{
			std::lock_guard lock(mutex_);
			return messages_;
		}

		/// The messages whose source is exactly this one.
		[[nodiscard]] std::vector<Message> from(const std::string& source) const
		{
			std::vector<Message> result;
			for (const Message& m : messages())
				if (m.source == source)
					result.push_back(m);
			return result;
		}

	private:
		mutable std::mutex mutex_;
		std::vector<Message> messages_;
	};

}

TEST(Log, DefaultLevelIsInfoAndNamesAreStable)
{
	EXPECT_EQ(kLevelAtStart, LogLevel::Info);
	EXPECT_EQ(log_level(), LogLevel::Info);
	EXPECT_STREQ(to_string(LogLevel::Error), "Error");
	EXPECT_STREQ(to_string(LogLevel::Warning), "Warning");
	EXPECT_STREQ(to_string(LogLevel::Info), "Info");
	EXPECT_TRUE(log_enabled(LogLevel::Error));
	EXPECT_TRUE(log_enabled(LogLevel::Warning));
	EXPECT_TRUE(log_enabled(LogLevel::Info));
}

TEST(Log, LevelFiltersMessages)
{
	LogCapture capture;

	set_log_level(LogLevel::Warning);
	EXPECT_EQ(log_level(), LogLevel::Warning);
	EXPECT_TRUE(log_enabled(LogLevel::Error));
	EXPECT_TRUE(log_enabled(LogLevel::Warning));
	EXPECT_FALSE(log_enabled(LogLevel::Info));
	GVL_LOG_ERROR(Alpha, "e");
	GVL_LOG_WARNING(Alpha, "w");
	GVL_LOG_INFO(Alpha, "i");
	ASSERT_EQ(capture.messages().size(), 2u);
	EXPECT_EQ(capture.messages()[0].level, LogLevel::Error);
	EXPECT_EQ(capture.messages()[1].level, LogLevel::Warning);

	set_log_level(LogLevel::Error);
	GVL_LOG_ERROR(Alpha, "e");
	GVL_LOG_WARNING(Alpha, "w");
	GVL_LOG_INFO(Alpha, "i");
	EXPECT_EQ(capture.messages().size(), 3u); // only the error was added

	// a call of log_message() itself is filtered too
	set_log_level(LogLevel::Error);
	log_message(LogLevel::Info, "Direct::source", "dropped");
	EXPECT_EQ(capture.messages().size(), 3u);
	log_message(LogLevel::Error, "Direct::source", "kept");
	ASSERT_EQ(capture.messages().size(), 4u);
	EXPECT_EQ(capture.messages()[3].source, "Direct::source");
	EXPECT_EQ(capture.messages()[3].text, "kept");

	set_log_level(LogLevel::Info);
	GVL_LOG_ERROR(Alpha, "e");
	GVL_LOG_WARNING(Alpha, "w");
	GVL_LOG_INFO(Alpha, "i");
	EXPECT_EQ(capture.messages().size(), 7u);
}

TEST(Log, HandlerGetsLevelSourceAndText)
{
	LogCapture capture;
	const int value = 5;
	GVL_LOG_WARNING(Alpha, "value " << value << " of " << 7.5 << ", " << std::string("text"));
	const auto messages = capture.messages();
	ASSERT_EQ(messages.size(), 1u);
	EXPECT_EQ(messages[0].level, LogLevel::Warning);
	EXPECT_EQ(messages[0].source, std::string("Alpha") + "::" + "TestBody");
	EXPECT_EQ(messages[0].text, "value 5 of 7.5, text");
}

TEST(Log, TextIsOnlyBuiltIfTheLevelIsEnabled)
{
	LogCapture capture;
	int built = 0;
	auto expensive = [&] { ++built; return 1; };

	set_log_level(LogLevel::Error);
	GVL_LOG_INFO(Alpha, "x" << expensive());
	GVL_LOG_WARNING(Alpha, "x" << expensive());
	EXPECT_EQ(built, 0);
	GVL_LOG_ERROR(Alpha, "x" << expensive());
	EXPECT_EQ(built, 1);
	EXPECT_EQ(capture.messages().size(), 1u);
}

TEST(Log, MacroIsOneStatement)
{
	LogCapture capture;
	if (capture.messages().empty())
		GVL_LOG_INFO(Alpha, "then");
	else
		GVL_LOG_INFO(Alpha, "else");
	ASSERT_EQ(capture.messages().size(), 1u);
	EXPECT_EQ(capture.messages()[0].text, "then");
}

TEST(Log, DefaultOutputIsStderrAndAHandlerReplacesIt)
{
	set_log_level(LogLevel::Info);
	set_log_handler({});

	testing::internal::CaptureStderr();
	GVL_LOG_WARNING(Alpha, "to stderr " << 3);
	const std::string out = testing::internal::GetCapturedStderr();
	EXPECT_EQ(out, "[gvl] Warning Alpha::TestBody: to stderr 3\n");

	{
		LogCapture capture;
		testing::internal::CaptureStderr();
		GVL_LOG_WARNING(Alpha, "to the handler");
		EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
		EXPECT_EQ(capture.messages().size(), 1u);
	}

	// an empty handler brings the default output back; the level is filtered there too
	testing::internal::CaptureStderr();
	set_log_level(LogLevel::Error);
	GVL_LOG_WARNING(Alpha, "filtered");
	GVL_LOG_ERROR(Alpha, "kept");
	set_log_level(LogLevel::Info);
	EXPECT_EQ(testing::internal::GetCapturedStderr(), "[gvl] Error Alpha::TestBody: kept\n");
}

TEST(Log, ManyThreadsLogWithoutLosingMessages)
{
	LogCapture capture;
	constexpr int kThreads = 6;
	constexpr int kPerThread = 300;
	std::vector<std::thread> threads;
	for (int t = 0; t < kThreads; ++t)
		threads.emplace_back([t] {
			for (int i = 0; i < kPerThread; ++i)
				GVL_LOG_INFO(Worker, "thread " << t << " message " << i);
		});
	for (auto& t : threads)
		t.join();
	const auto messages = capture.messages();
	EXPECT_EQ(messages.size(), static_cast<std::size_t>(kThreads * kPerThread));
	for (const Message& m : messages)
	{
		EXPECT_EQ(m.level, LogLevel::Info);
		EXPECT_EQ(m.text.rfind("thread ", 0), 0u);
	}
}

TEST(Log, HandlerCanBeReplacedWhileOtherThreadsLog)
{
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> received{ 0 };
	std::thread logger([&] {
		while (!stop)
			GVL_LOG_INFO(Worker, "x");
	});
	for (int i = 0; i < 200; ++i)
	{
		set_log_handler([&](LogLevel, std::string_view, std::string_view) { ++received; });
		set_log_handler({});
	}
	stop = true;
	logger.join();

	// whatever the loop above delivered, a handler that is set now receives every further message and the old one none
	const uint64_t before = received.load();
	std::atomic<uint64_t> later{ 0 };
	set_log_handler([&](LogLevel, std::string_view, std::string_view) { ++later; });
	GVL_LOG_INFO(Worker, "one");
	GVL_LOG_INFO(Worker, "two");
	set_log_handler({});
	EXPECT_EQ(later.load(), 2u);
	EXPECT_EQ(received.load(), before);
}

// ---- the messages of the classes (they were icl_core LOGGING_*_C(VulkanLog, ...) before) ----

TEST_F(VulkanTest, BitMapReportsPointsOutsideAsWarning)
{
	BitVoxelMap map(ctx(), { 4, 3, 2 }, 1.0f);
	LogCapture capture;
	const std::vector<Point3f> points{ { 0.5f, 0.5f, 0.5f }, { 40.f, 0.5f, 0.5f }, { 0.5f, -3.f, 0.5f } };
	EXPECT_EQ(map.insert_point_cloud(points, eBVM_OCCUPIED), 2u);
	const auto messages = capture.from("BitVoxelMap::insert_point_cloud");
	ASSERT_EQ(messages.size(), 1u);
	EXPECT_EQ(messages[0].level, LogLevel::Warning);
	EXPECT_EQ(messages[0].text, "2 point(s) lie outside the map dimensions!");
	EXPECT_EQ(capture.messages().size(), 1u);
}

TEST_F(VulkanTest, FacadeReportsUnknownAndUsedNamesAsErrors)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 8, 8, 8 }, 0.1f);
	LogCapture capture;

	EXPECT_FALSE(gvl.del_map("nope"));
	ASSERT_EQ(capture.from("GpuVoxels::del_map").size(), 1u);
	EXPECT_EQ(capture.from("GpuVoxels::del_map")[0].level, LogLevel::Error);
	EXPECT_EQ(capture.from("GpuVoxels::del_map")[0].text, "Map with name 'nope' not found.");

	ASSERT_TRUE(gvl.add_map(MT_BITVECTOR_VOXELMAP, "a"));
	EXPECT_FALSE(gvl.add_map(MT_BITVECTOR_VOXELMAP, "a"));
	ASSERT_EQ(capture.from("GpuVoxels::add_map").size(), 1u);
	EXPECT_EQ(capture.from("GpuVoxels::add_map")[0].level, LogLevel::Error);
	EXPECT_EQ(capture.from("GpuVoxels::add_map")[0].text, "Map with name 'a' already exists.");
}

TEST_F(VulkanTest, ContextReportsItsDeviceAsInfo)
{
	LogCapture capture;
	ContextCreateInfo info;
	info.require_validation = std::getenv("GVL_ALLOW_NO_VALIDATION") == nullptr;
	Context second(info);
	const auto messages = capture.from("Context::Context");
	ASSERT_EQ(messages.size(), 1u);
	EXPECT_EQ(messages[0].level, LogLevel::Info);
	EXPECT_EQ(messages[0].text.rfind("Vulkan device '" + second.info().name + "' (", 0), 0u) << messages[0].text;
	EXPECT_NE(messages[0].text.find(", validation "), std::string::npos);

	// at the level Warning the line is not delivered
	set_log_level(LogLevel::Warning);
	Context third(info);
	EXPECT_EQ(capture.from("Context::Context").size(), 1u);
}
