// Slice S3: sensor data insertion with ray casting into the Vulkan ProbVoxelMap against the independent CPU reference.
#include "vulkan_fixture.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>

#include "prob_voxel_map_support.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
using namespace gvl_test::prob;

namespace
{
	UpdateOptions with_mode(UpdateMode mode, SensorModel model = {}) { return UpdateOptions{ model, mode }; }

	/// Inserts one scan into the map and the reference and checks the returned counters.
	void insert_scan(ProbVoxelMap& map, ProbVoxelMapReference& ref, const std::vector<Point3f>& points, Point3f origin, bool raycast,
		UpdateMode mode, const BitVoxelMap* robot = nullptr, const BitVoxelMapReference* robot_ref = nullptr, uint8_t robot_bit = eBVM_OCCUPIED,
		SensorModel model = {})
	{
		const float o[3] = { origin.x, origin.y, origin.z };
		const auto generated = ref.sensor_events(points, o, raycast, robot_ref, robot_bit, model.free_update, model.occupied_update);
		const auto result = map.insert_sensor_data(points, origin, raycast, robot, robot_bit, with_mode(mode, model));
		EXPECT_EQ(result.outside, generated.outside) << mode_name(mode);
		EXPECT_EQ(result.cut_by_robot, generated.cut_by_robot) << mode_name(mode);
		apply_to_reference(ref, generated.events, mode);
	}

	int8_t value_at(const ProbVoxelMap& map, uint64_t index) { return map.download_occupancy(index, 1).front(); }
}

TEST(RayWalk, ReferenceWalkHasTheClaimedGeometricProperties)
{
	// The shader relies on this: the CUDA walk visits 1 + dx + dy + dz voxels, starts at `from`, ends at `to`, moves one
	// axis by one voxel per step towards the target and never leaves the box of its end points (so it never leaves the map).
	auto check = [](std::array<uint32_t, 3> from, std::array<uint32_t, 3> to) -> ::testing::AssertionResult
	{
		const auto path = ProbVoxelMapReference::walk(from, to);
		int64_t expected_length = 1;
		for (int a = 0; a < 3; ++a)
			expected_length += from[a] > to[a] ? from[a] - to[a] : to[a] - from[a];
		if (static_cast<int64_t>(path.size()) != expected_length)
			return ::testing::AssertionFailure() << "length " << path.size() << " expected " << expected_length;
		for (int a = 0; a < 3; ++a)
			if (path.front()[a] != from[a] || path.back()[a] != to[a])
				return ::testing::AssertionFailure() << "wrong end points";
		for (std::size_t i = 0; i < path.size(); ++i)
		{
			for (int a = 0; a < 3; ++a)
			{
				const int64_t lo = std::min(from[a], to[a]), hi = std::max(from[a], to[a]);
				if (path[i][a] < lo || path[i][a] > hi)
					return ::testing::AssertionFailure() << "step " << i << " leaves the box on axis " << a;
			}
			if (i == 0)
				continue;
			int changed = 0;
			for (int a = 0; a < 3; ++a)
			{
				const int64_t delta = path[i][a] - path[i - 1][a];
				if (delta != 0)
				{
					++changed;
					if (delta != (to[a] > from[a] ? 1 : -1))
						return ::testing::AssertionFailure() << "step " << i << " moves away from the target on axis " << a;
				}
			}
			if (changed != 1)
				return ::testing::AssertionFailure() << "step " << i << " changes " << changed << " axes";
		}
		return ::testing::AssertionSuccess();
	};

	for (uint32_t fx = 0; fx < 7; ++fx)
		for (uint32_t fy = 0; fy < 7; ++fy)
			for (uint32_t fz = 0; fz < 7; ++fz)
				for (uint32_t tx = 0; tx < 7; ++tx)
					for (uint32_t ty = 0; ty < 7; ++ty)
						for (uint32_t tz = 0; tz < 7; ++tz)
							ASSERT_TRUE(check({ fx, fy, fz }, { tx, ty, tz })) << fx << fy << fz << " -> " << tx << ty << tz;

	std::mt19937 rng(71);
	std::uniform_int_distribution<uint32_t> any(0, 400);
	for (int i = 0; i < 3000; ++i)
	{
		const std::array<uint32_t, 3> from{ any(rng), any(rng), any(rng) }, to{ any(rng), any(rng), any(rng) };
		ASSERT_TRUE(check(from, to)) << i;
	}
	// strongly anisotropic rays
	ASSERT_TRUE(check({ 0, 0, 0 }, { 5000, 3, 1 }));
	ASSERT_TRUE(check({ 5000, 3, 1 }, { 0, 0, 0 }));
	ASSERT_TRUE(check({ 2, 4000, 9 }, { 2, 0, 9 }));
}

TEST_F(VulkanTest, ProbSensorRaysVisitTheWalkVoxelsInEveryMode)
{
	const Dim3ui dims{ 12, 9, 7 };
	const std::vector<std::pair<Point3f, Point3f>> rays{
		{ { 0.55f, 0.55f, 0.55f }, { 1.15f, 0.85f, 0.65f } },     // short diagonal
		{ { 0.55f, 0.55f, 0.55f }, { 0.55f, 0.55f, 0.55f } },     // origin voxel == hit voxel
		{ { 0.05f, 0.05f, 0.05f }, { 1.15f, 0.85f, 0.65f } },     // full diagonal
		{ { 1.15f, 0.85f, 0.65f }, { 0.05f, 0.05f, 0.05f } },     // reversed direction
		{ { 0.55f, 0.45f, 0.35f }, { 1.15f, 0.45f, 0.35f } },     // along x
		{ { 0.55f, 0.85f, 0.35f }, { 0.55f, 0.05f, 0.35f } },     // along -y
		{ { 0.55f, 0.45f, 0.65f }, { 0.55f, 0.45f, 0.05f } },     // along -z
		{ { 0.05f, 0.85f, 0.05f }, { 1.15f, 0.05f, 0.65f } },     // mixed signs
	};
	for (UpdateMode mode : { UpdateMode::Atomic, UpdateMode::Accumulate, UpdateMode::Sequential })
		for (const auto& [origin, hit] : rays)
		{
			ProbVoxelMap map(ctx(), dims, kSide);
			auto ref = prob_reference(dims);
			insert_scan(map, ref, { hit }, origin, true, mode);
			EXPECT_TRUE(equal_to_reference(map, ref)) << mode_name(mode);
			uint32_t known = 0;
			for (int8_t v : map.download_occupancy())
				known += v != -128 ? 1 : 0;
			EXPECT_GE(known, 1u);
		}
	// a single known ray: hit voxel gets -10 + 72, the others on the ray -10
	ProbVoxelMap map(ctx(), { 5, 1, 1 }, 1.0f);
	map.insert_sensor_data(std::vector<Point3f>{ { 3.5f, 0.5f, 0.5f } }, { 0.5f, 0.5f, 0.5f }, true);
	EXPECT_EQ(map.download_occupancy(), (std::vector<int8_t>{ -10, -10, -10, 62, -128 }));
}

TEST_F(VulkanTest, ProbSensorScansMatchTheReference)
{
	std::mt19937 rng(72);
	const Dim3ui dims = kProbDims;
	for (UpdateMode mode : { UpdateMode::Accumulate, UpdateMode::Sequential })
		for (bool raycast : { true, false })
		{
			ProbVoxelMap map(ctx(), dims, kSide);
			auto ref = prob_reference(dims);
			for (int scan = 0; scan < 4; ++scan) // several scans: the accumulators must be clean between calls
			{
				const Point3f origin = random_inside_point(rng, dims, kSide);
				insert_scan(map, ref, random_points(rng, 1500, dims), origin, raycast, mode);
				ASSERT_TRUE(equal_to_reference(map, ref)) << mode_name(mode) << " raycast " << raycast << " scan " << scan;
			}
		}
}

TEST_F(VulkanTest, ProbSensorAtomicModeAgreesWhereNoOrderEffectIsPossible)
{
	std::mt19937 rng(73);
	const Dim3ui dims = kProbDims;
	ProbVoxelMap map(ctx(), dims, kSide);
	auto ref = prob_reference(dims);
	const auto points = random_points(rng, 1500, dims);
	const Point3f origin = random_inside_point(rng, dims, kSide);
	const float o[3] = { origin.x, origin.y, origin.z };
	const auto generated = ref.sensor_events(points, o, true, nullptr, 0, -10, 72);
	map.insert_sensor_data(points, origin, true, nullptr, eBVM_OCCUPIED, with_mode(UpdateMode::Atomic));
	ref.apply_sequential(generated.events);

	// A voxel whose updates have an absolute sum of at most 127 can never saturate: every serial order gives one result.
	std::map<uint64_t, long long> absolute_sum;
	for (const auto& e : generated.events)
		absolute_sum[e.voxel] += std::abs(e.increment);
	const auto gpu = map.download_occupancy();
	uint32_t compared = 0, saturating = 0;
	for (std::size_t i = 0; i < gpu.size(); ++i)
	{
		ASSERT_GE(gpu[i], -128);
		const auto it = absolute_sum.find(i);
		if (it == absolute_sum.end())
		{
			ASSERT_EQ(gpu[i], -128) << "untouched voxels stay unknown";
			continue;
		}
		ASSERT_NE(gpu[i], -128) << "touched voxels become known";
		if (it->second <= 127)
		{
			ASSERT_EQ(gpu[i], ref.get(i)) << "voxel " << i;
			++compared;
		}
		else
			++saturating;
	}
	EXPECT_GT(compared, 100u);
	EXPECT_GT(saturating, 0u) << "the scan must contain voxels where the order can matter";
}

TEST_F(VulkanTest, ProbSensorWithoutRaycastingOnlyUpdatesTheHitVoxels)
{
	std::mt19937 rng(74);
	const auto points = random_points(rng, 800, kProbDims);
	const Point3f origin = random_inside_point(rng, kProbDims, kSide);
	for (UpdateMode mode : { UpdateMode::Atomic, UpdateMode::Accumulate, UpdateMode::Sequential })
	{
		// all updates are +72: saturation is order independent, so every mode must equal the reference
		ProbVoxelMap map(ctx(), kProbDims, kSide);
		auto ref = prob_reference();
		insert_scan(map, ref, points, origin, false, mode);
		EXPECT_TRUE(equal_to_reference(map, ref)) << mode_name(mode);
		for (int8_t v : map.download_occupancy())
			ASSERT_TRUE(v == -128 || v >= 72) << "only hit voxels may be known";
	}
}

TEST_F(VulkanTest, ProbSensorDropsMeasurementsWithOriginOrHitOutsideTheMap)
{
	const Dim3ui dims{ 12, 9, 7 };
	const float nan = std::numeric_limits<float>::quiet_NaN();
	const std::vector<Point3f> inside{ { 0.55f, 0.45f, 0.35f }, { 1.05f, 0.75f, 0.55f } };
	const std::vector<Point3f> mixed{ { 0.55f, 0.45f, 0.35f }, { -0.05f, 0.5f, 0.5f }, { 1.25f, 0.5f, 0.5f }, { 0.5f, 0.95f, 0.5f },
		{ 0.5f, 0.5f, 0.75f }, { nan, 0.5f, 0.5f }, { 0.5f, nan, 0.5f }, { 1.05f, 0.75f, 0.55f } };
	for (UpdateMode mode : { UpdateMode::Atomic, UpdateMode::Accumulate, UpdateMode::Sequential })
	{
		// origin outside: everything is dropped, the map stays untouched
		for (Point3f origin : { Point3f{ -0.05f, 0.5f, 0.5f }, Point3f{ 0.5f, 0.95f, 0.5f }, Point3f{ 0.5f, 0.5f, 0.75f }, Point3f{ nan, 0.5f, 0.5f },
			Point3f{ 5.0f, 5.0f, 5.0f } })
		{
			ProbVoxelMap map(ctx(), dims, kSide);
			const auto result = map.insert_sensor_data(inside, origin, true, nullptr, eBVM_OCCUPIED, with_mode(mode));
			EXPECT_EQ(result.outside, 2u) << mode_name(mode);
			for (int8_t v : map.download_occupancy())
				ASSERT_EQ(v, -128);
			// also without ray casting: CUDA still requires the sensor voxel to be inside
			const auto result2 = map.insert_sensor_data(inside, origin, false, nullptr, eBVM_OCCUPIED, with_mode(mode));
			EXPECT_EQ(result2.outside, 2u) << mode_name(mode);
		}
		// hits outside / NaN: only those are dropped
		ProbVoxelMap map(ctx(), dims, kSide);
		auto ref = prob_reference(dims);
		insert_scan(map, ref, mixed, { 0.55f, 0.55f, 0.55f }, true, mode);
		EXPECT_TRUE(equal_to_reference(map, ref)) << mode_name(mode);
		const float o[3] = { 0.55f, 0.55f, 0.55f };
		EXPECT_EQ(ref.sensor_events(mixed, o, true, nullptr, 0, -10, 72).outside, 6u);
	}
}

TEST_F(VulkanTest, ProbSensorCutsTheRobotOutOfTheMeasurements)
{
	std::mt19937 rng(75);
	const Dim3ui dims = kProbDims;
	BitVoxelMap robot(ctx(), dims, kSide);
	auto robot_ref = map_reference(dims);
	const auto robot_points = random_points(rng, 700, dims);
	robot.insert_point_cloud(robot_points, eBVM_OCCUPIED);
	robot_ref.insert_points(robot_points, eBVM_OCCUPIED);
	const auto other_bit_points = random_points(rng, 700, dims);
	robot.insert_point_cloud(other_bit_points, 40);
	robot_ref.insert_points(other_bit_points, 40);

	for (uint8_t bit : { uint8_t{ eBVM_OCCUPIED }, uint8_t{ 40 }, uint8_t{ 99 } })
		for (UpdateMode mode : { UpdateMode::Accumulate, UpdateMode::Sequential })
		{
			ProbVoxelMap map(ctx(), dims, kSide);
			auto ref = prob_reference(dims);
			auto scan = random_points(rng, 1500, dims);
			scan.insert(scan.end(), robot_points.begin(), robot_points.begin() + 200); // certainly cut for bit 1
			insert_scan(map, ref, scan, random_inside_point(rng, dims, kSide), true, mode, &robot, &robot_ref, bit);
			EXPECT_TRUE(equal_to_reference(map, ref)) << mode_name(mode) << " bit " << int(bit);
		}

	// cut counts: bit 99 is set nowhere, bit 1 and 40 somewhere
	ProbVoxelMap map(ctx(), dims, kSide);
	const auto none = map.insert_sensor_data(robot_points, { 0.5f, 0.5f, 0.5f }, false, &robot, 99);
	EXPECT_EQ(none.cut_by_robot, 0u);
	map.clear_map();
	const auto all = map.insert_sensor_data(robot_points, { 0.5f, 0.5f, 0.5f }, false, &robot, eBVM_OCCUPIED);
	EXPECT_EQ(all.cut_by_robot, static_cast<uint32_t>(robot_points.size())) << "every point lies in a robot voxel";
	for (int8_t v : map.download_occupancy())
		ASSERT_EQ(v, -128) << "nothing is inserted when everything is cut";

	BitVoxelMap wrong(ctx(), { 5, 5, 5 }, kSide);
	EXPECT_THROW((void)map.insert_sensor_data(robot_points, { 0.5f, 0.5f, 0.5f }, true, &wrong), VulkanError);
}

TEST_F(VulkanTest, ProbSensorProcessesMorePointsThanTheMapHasVoxels)
{
	// CUDA stops after min(points, voxels) points; the port processes all of them
	const Dim3ui dims{ 4, 4, 4 };
	std::mt19937 rng(76);
	const auto points = random_points(rng, 600, dims, 1.0f);
	for (UpdateMode mode : { UpdateMode::Accumulate, UpdateMode::Sequential })
	{
		ProbVoxelMap map(ctx(), dims, 1.0f);
		auto ref = prob_reference(dims, 1.0f);
		insert_scan(map, ref, points, { 2.5f, 2.5f, 2.5f }, true, mode);
		EXPECT_TRUE(equal_to_reference(map, ref)) << mode_name(mode);
		int hits = 0;
		for (int8_t v : map.download_occupancy())
			hits += v != -128 ? 1 : 0;
		EXPECT_GT(hits, 40) << "the points beyond the 64th one matter";
	}
}

TEST_F(VulkanTest, ProbSensorSequentialModeSplitsHugeBatchesIntoChunks)
{
	const Dim3ui dims{ 96, 96, 96 };
	std::mt19937 rng(77);
	const auto points = random_points(rng, 80000, dims);
	const Point3f origin{ 4.85f, 4.75f, 4.65f };
	ProbVoxelMap map(ctx(), dims, kSide);
	auto ref = prob_reference(dims);
	const float o[3] = { origin.x, origin.y, origin.z };
	const auto generated = ref.sensor_events(points, o, true, nullptr, 0, -10, 72);
	ASSERT_GT(generated.events.size(), static_cast<std::size_t>(ProbVoxelMap::kMaxSequentialEvents)) << "must need more than one chunk";
	const auto result = map.insert_sensor_data(points, origin, true, nullptr, eBVM_OCCUPIED, with_mode(UpdateMode::Sequential));
	EXPECT_EQ(result.outside, 0u);
	ref.apply_sequential(generated.events);
	EXPECT_TRUE(equal_to_reference(map, ref));
}

TEST_F(VulkanTest, ProbSensorSequentialModeAppliesSmallChunksInOrder)
{
	// a tiny chunk limit forces many chunks; chunk borders fall inside the runs of updates of hot voxels
	std::mt19937 rng(79);
	const Dim3ui dims = kProbDims;
	for (uint32_t limit : { 1000u, 777u, 205u })
	{
		ProbVoxelMap map(ctx(), dims, kSide);
		auto ref = prob_reference(dims);
		UpdateOptions options = with_mode(UpdateMode::Sequential);
		options.max_sequential_events = limit;
		const auto points = random_points(rng, 1200, dims);
		const Point3f origin = random_inside_point(rng, dims, kSide);
		const float o[3] = { origin.x, origin.y, origin.z };
		const auto generated = ref.sensor_events(points, o, true, nullptr, 0, -10, 72);
		ASSERT_GT(generated.events.size(), 5u * limit) << "several chunks are needed";
		map.insert_sensor_data(points, origin, true, nullptr, eBVM_OCCUPIED, options);
		ref.apply_sequential(generated.events);
		EXPECT_TRUE(equal_to_reference(map, ref)) << "limit " << limit;
	}
}

TEST_F(VulkanTest, ProbSensorSequentialModeRejectsRaysLongerThanTheChunkLimit)
{
	ProbVoxelMap map(ctx(), { 64, 1, 1 }, 1.0f);
	UpdateOptions options = with_mode(UpdateMode::Sequential);
	options.max_sequential_events = 40;
	const std::vector<Point3f> far{ { 49.5f, 0.5f, 0.5f } };   // 50 free updates + 1 hit
	const std::vector<Point3f> near{ { 29.5f, 0.5f, 0.5f } };  // 30 + 1
	EXPECT_THROW((void)map.insert_sensor_data(far, { 0.5f, 0.5f, 0.5f }, true, nullptr, eBVM_OCCUPIED, options), VulkanError);
	EXPECT_EQ(value_at(map, 0), -128) << "nothing was applied";
	EXPECT_EQ(map.insert_sensor_data(near, { 0.5f, 0.5f, 0.5f }, true, nullptr, eBVM_OCCUPIED, options).outside, 0u);
	EXPECT_EQ(value_at(map, 29), 62);
	// the same far ray without ray casting is a single update
	EXPECT_EQ(map.insert_sensor_data(far, { 0.5f, 0.5f, 0.5f }, false, nullptr, eBVM_OCCUPIED, options).outside, 0u);
	EXPECT_EQ(value_at(map, 49), 72);
	options.max_sequential_events = 0;
	EXPECT_THROW((void)map.insert_sensor_data(near, { 0.5f, 0.5f, 0.5f }, true, nullptr, eBVM_OCCUPIED, options), VulkanError);
	options.max_sequential_events = ProbVoxelMap::kMaxSequentialEvents + 1;
	EXPECT_THROW((void)map.insert_sensor_data(near, { 0.5f, 0.5f, 0.5f }, true, nullptr, eBVM_OCCUPIED, options), VulkanError);
}

TEST_F(VulkanTest, ProbSensorCustomModelIsUsedForFreeAndHitUpdates)
{
	const SensorModel model{ -3, 5 };
	for (UpdateMode mode : { UpdateMode::Atomic, UpdateMode::Accumulate, UpdateMode::Sequential })
	{
		ProbVoxelMap map(ctx(), { 5, 1, 1 }, 1.0f);
		map.insert_sensor_data(std::vector<Point3f>{ { 3.5f, 0.5f, 0.5f } }, { 0.5f, 0.5f, 0.5f }, true, nullptr, eBVM_OCCUPIED,
			with_mode(mode, model));
		EXPECT_EQ(map.download_occupancy(), (std::vector<int8_t>{ -3, -3, -3, 2, -128 })) << mode_name(mode);
	}
}

TEST_F(VulkanTest, ProbSensorPerformanceReport)
{
	const Dim3ui dims{ 128, 128, 128 };
	std::mt19937 rng(78);
	const auto points = random_points(rng, 100000, dims);
	const Point3f origin{ 6.4f, 6.4f, 6.4f };
	for (UpdateMode mode : { UpdateMode::Atomic, UpdateMode::Accumulate, UpdateMode::Sequential })
	{
		ProbVoxelMap map(ctx(), dims, kSide);
		const auto t0 = std::chrono::steady_clock::now();
		const auto result = map.insert_sensor_data(points, origin, true, nullptr, eBVM_OCCUPIED, with_mode(mode));
		const auto t1 = std::chrono::steady_clock::now();
		EXPECT_EQ(result.outside, 0u);
		std::cout << "[perf] 100k rays into 128^3, " << mode_name(mode) << ": "
			<< std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
	}
}
