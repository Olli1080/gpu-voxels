// Slice P2: data-parallel primitives (scan, stable radix sort) against the C++ standard library.
#include "vulkan_fixture.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/Primitives.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;

namespace
{
	std::vector<uint32_t> random_values(std::mt19937& rng, std::size_t n, uint32_t max_value)
	{
		std::uniform_int_distribution<uint32_t> dist(0, max_value);
		std::vector<uint32_t> v(n);
		for (auto& x : v)
			x = dist(rng);
		return v;
	}

	::testing::AssertionResult sorted_like_reference(Primitives& primitives, std::vector<uint32_t> keys, uint32_t key_bits)
	{
		std::vector<uint32_t> values(keys.size());
		std::iota(values.begin(), values.end(), 0u); // original index: makes stability observable

		const uint32_t mask = key_bits >= 32 ? 0xFFFFFFFFu : ((1u << key_bits) - 1u);
		std::vector<std::pair<uint32_t, uint32_t>> expected;
		for (std::size_t i = 0; i < keys.size(); ++i)
			expected.emplace_back(keys[i], values[i]);
		std::stable_sort(expected.begin(), expected.end(), [&](const auto& a, const auto& b) { return (a.first & mask) < (b.first & mask); });

		primitives.sort_pairs(keys, values, key_bits);
		for (std::size_t i = 0; i < keys.size(); ++i)
			if (keys[i] != expected[i].first || values[i] != expected[i].second)
				return ::testing::AssertionFailure() << "n=" << keys.size() << " bits=" << key_bits << " mismatch at " << i << ": got ("
					<< keys[i] << ", " << values[i] << ") expected (" << expected[i].first << ", " << expected[i].second << ")";
		return ::testing::AssertionSuccess();
	}
}

TEST_F(VulkanTest, ScanMatchesStdExclusiveScan)
{
	Primitives primitives(ctx());
	std::mt19937 rng(40);
	for (std::size_t n : { 1u, 2u, 3u, 255u, 256u, 257u, 511u, 512u, 513u, 1023u, 1024u, 1025u, 262144u, 262145u, 1'000'003u })
	{
		for (uint32_t max_value : { 3u, 0xFFFFFFFFu }) // small values and values that wrap around
		{
			const auto input = random_values(rng, n, max_value);
			std::vector<uint32_t> expected(n);
			std::exclusive_scan(input.begin(), input.end(), expected.begin(), 0u);
			EXPECT_EQ(primitives.exclusive_scan(input), expected) << "n=" << n << " max=" << max_value;
		}
	}
	EXPECT_TRUE(primitives.exclusive_scan({}).empty());
}

TEST_F(VulkanTest, ScanOfFiveMillionElements)
{
	Primitives primitives(ctx());
	std::vector<uint32_t> input(5'000'000, 1u);
	const auto t0 = std::chrono::steady_clock::now();
	const auto result = primitives.exclusive_scan(input);
	const auto t1 = std::chrono::steady_clock::now();
	std::cout << "[vulkan] scan of 5 M elements incl. transfers: " << std::chrono::duration<double, std::milli>(t1 - t0).count()
		<< " ms (" << ctx().info().name << ")" << std::endl;
	for (std::size_t i = 0; i < result.size(); i += 4999)
		ASSERT_EQ(result[i], static_cast<uint32_t>(i));
	EXPECT_EQ(result.back(), 4'999'999u);
}

TEST_F(VulkanTest, SortPairsMatchesStableSort)
{
	Primitives primitives(ctx());
	std::mt19937 rng(41);
	for (std::size_t n : { 2u, 3u, 255u, 256u, 257u, 1000u, 65537u, 300000u })
	{
		EXPECT_TRUE(sorted_like_reference(primitives, random_values(rng, n, 0xFFFFFFFFu), 32)) << "random 32 bit";
		EXPECT_TRUE(sorted_like_reference(primitives, random_values(rng, n, 15u), 32)) << "many duplicates";
		EXPECT_TRUE(sorted_like_reference(primitives, std::vector<uint32_t>(n, 7u), 32)) << "all equal";
		std::vector<uint32_t> ascending(n);
		std::iota(ascending.begin(), ascending.end(), 0u);
		EXPECT_TRUE(sorted_like_reference(primitives, ascending, 32)) << "already sorted";
		std::reverse(ascending.begin(), ascending.end());
		EXPECT_TRUE(sorted_like_reference(primitives, ascending, 32)) << "reverse sorted";
	}
}

TEST_F(VulkanTest, SortPairsRespectsKeyBits)
{
	Primitives primitives(ctx());
	std::mt19937 rng(42);
	// 1 pass, 2 passes (odd/even number of ping-pong steps), 3 passes; high bits must not influence the order
	for (uint32_t bits : { 1u, 5u, 8u, 9u, 13u, 16u, 17u, 24u, 25u, 31u })
		EXPECT_TRUE(sorted_like_reference(primitives, random_values(rng, 50'000, 0xFFFFFFFFu), bits)) << "bits " << bits;
	// keys that really fit into the bits
	EXPECT_TRUE(sorted_like_reference(primitives, random_values(rng, 70'000, (1u << 21) - 1), 21));
}

TEST_F(VulkanTest, SortOfTwoMillionPairs)
{
	Primitives primitives(ctx());
	std::mt19937 rng(43);
	auto keys = random_values(rng, 2'000'000, (1u << 24) - 1);
	std::vector<uint32_t> values(keys.size());
	std::iota(values.begin(), values.end(), 0u);
	const auto original = keys;
	const auto t0 = std::chrono::steady_clock::now();
	primitives.sort_pairs(keys, values, 24);
	const auto t1 = std::chrono::steady_clock::now();
	std::cout << "[vulkan] radix sort of 2 M pairs (24 bit keys) incl. transfers: "
		<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms (" << ctx().info().name << ")" << std::endl;
	EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
	for (std::size_t i = 0; i < keys.size(); i += 997)
		ASSERT_EQ(original[values[i]], keys[i]) << "value must still belong to its key";
}

TEST_F(VulkanTest, PrimitivesRejectInvalidArguments)
{
	Primitives primitives(ctx());
	std::vector<uint32_t> keys(4), values(3);
	EXPECT_THROW(primitives.sort_pairs(keys, values), VulkanError);
	std::vector<uint32_t> same(4);
	EXPECT_THROW(primitives.sort_pairs(keys, same, 0), VulkanError);
	EXPECT_THROW(primitives.sort_pairs(keys, same, 33), VulkanError);
	primitives.sort_pairs(keys, same, 32); // all zero input is fine
}

TEST_F(VulkanTest, IotaWritesTheIndexOfEveryElement)
{
	Primitives primitives(ctx());
	for (uint32_t n : { 1u, 2u, 255u, 256u, 257u, 65'537u, 1'000'003u })
	{
		DeviceBuffer buffer(ctx(), static_cast<uint64_t>(n + 5) * sizeof(uint32_t));
		buffer.fill(0xDEADBEEFu); // the five elements behind n must stay untouched
		CommandBatch batch(ctx());
		primitives.record_iota(batch, buffer, n);
		batch.submit_and_wait();
		const auto values = buffer.download_vector<uint32_t>(n + 5);
		for (uint32_t i = 0; i < n; ++i)
			ASSERT_EQ(values[i], i) << "n=" << n;
		for (uint32_t i = n; i < n + 5; ++i)
			ASSERT_EQ(values[i], 0xDEADBEEFu) << "n=" << n;
	}

	// n = 0 records nothing, a too small buffer is rejected
	DeviceBuffer small(ctx(), 16);
	CommandBatch batch(ctx());
	primitives.record_iota(batch, small, 0);
	EXPECT_EQ(batch.dispatch_count(), 0u);
	EXPECT_THROW(primitives.record_iota(batch, small, 5), VulkanError);
}

namespace
{
	::testing::AssertionResult sorted64_like_reference(Primitives& primitives, std::vector<uint64_t> keys, uint32_t key_bits)
	{
		std::vector<uint32_t> values(keys.size());
		std::iota(values.begin(), values.end(), 0u);
		const uint64_t mask = key_bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << key_bits) - 1);
		std::vector<std::pair<uint64_t, uint32_t>> expected;
		for (std::size_t i = 0; i < keys.size(); ++i)
			expected.emplace_back(keys[i], values[i]);
		std::stable_sort(expected.begin(), expected.end(), [&](const auto& a, const auto& b) { return (a.first & mask) < (b.first & mask); });

		primitives.sort_pairs64(keys, values, key_bits);
		for (std::size_t i = 0; i < keys.size(); ++i)
			if (keys[i] != expected[i].first || values[i] != expected[i].second)
				return ::testing::AssertionFailure() << "n=" << keys.size() << " bits=" << key_bits << " mismatch at " << i << ": got ("
					<< keys[i] << ", " << values[i] << ") expected (" << expected[i].first << ", " << expected[i].second << ")";
		return ::testing::AssertionSuccess();
	}

	std::vector<uint64_t> random_keys64(std::mt19937_64& rng, std::size_t n, uint64_t max_value)
	{
		std::uniform_int_distribution<uint64_t> dist(0, max_value);
		std::vector<uint64_t> v(n);
		for (auto& x : v)
			x = dist(rng);
		return v;
	}
}

TEST_F(VulkanTest, SortPairs64MatchesStableSort)
{
	Primitives primitives(ctx());
	std::mt19937_64 rng(51);
	for (std::size_t n : { 2u, 3u, 255u, 256u, 257u, 1000u, 65537u, 200000u })
	{
		EXPECT_TRUE(sorted64_like_reference(primitives, random_keys64(rng, n, ~uint64_t(0)), 64)) << "random 64 bit";
		EXPECT_TRUE(sorted64_like_reference(primitives, random_keys64(rng, n, (uint64_t(1) << 42) - 1), 42)) << "Morton ids";
		EXPECT_TRUE(sorted64_like_reference(primitives, random_keys64(rng, n, 15), 64)) << "many duplicates, high word zero";
		auto high_only = random_keys64(rng, n, 15);
		for (auto& k : high_only)
			k <<= 32;
		EXPECT_TRUE(sorted64_like_reference(primitives, high_only, 64)) << "only the high word differs";
		EXPECT_TRUE(sorted64_like_reference(primitives, std::vector<uint64_t>(n, 0x1234'5678'9ABC'DEF0ull), 64)) << "all equal";
		std::vector<uint64_t> descending(n);
		std::iota(descending.rbegin(), descending.rend(), uint64_t(1) << 33);
		EXPECT_TRUE(sorted64_like_reference(primitives, descending, 64)) << "reverse sorted across the word boundary";
	}
}

TEST_F(VulkanTest, SortPairs64RespectsKeyBits)
{
	Primitives primitives(ctx());
	std::mt19937_64 rng(52);
	// digits inside the low word, exactly at the word boundary (32), inside the high word (odd / even pass counts)
	for (uint32_t bits : { 1u, 8u, 31u, 32u, 33u, 40u, 41u, 42u, 48u, 56u, 63u })
		EXPECT_TRUE(sorted64_like_reference(primitives, random_keys64(rng, 40'000, ~uint64_t(0)), bits)) << "bits " << bits;
}

TEST_F(VulkanTest, SortPairs64RejectsInvalidArguments)
{
	Primitives primitives(ctx());
	std::vector<uint64_t> keys(4);
	std::vector<uint32_t> values(3), same(4);
	EXPECT_THROW(primitives.sort_pairs64(keys, values), VulkanError);
	EXPECT_THROW(primitives.sort_pairs64(keys, same, 0), VulkanError);
	EXPECT_THROW(primitives.sort_pairs64(keys, same, 65), VulkanError);
	primitives.sort_pairs64(keys, same, 64);

	DeviceBuffer small_keys(ctx(), 4 * 8 - 4), values_buffer(ctx(), 16); // one word too small for 4 wide keys
	KeepAlive keep_alive;
	CommandBatch batch(ctx());
	EXPECT_THROW(primitives.record_sort_pairs64(batch, small_keys, values_buffer, 4, 64, keep_alive), VulkanError);
}
