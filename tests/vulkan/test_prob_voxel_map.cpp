// Slice S3: Vulkan ProbVoxelMap (insertion, update modes, collision) against the independent CPU reference.
#include "vulkan_fixture.h"

#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>

#include "prob_voxel_map_support.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
using namespace gvl_test::prob;

namespace
{
	constexpr UpdateMode kAllModes[] = { UpdateMode::Atomic, UpdateMode::Accumulate, UpdateMode::Sequential };

	UpdateOptions with_mode(UpdateMode mode, SensorModel model = {}) { return UpdateOptions{ model, mode }; }

	int8_t value_at(const ProbVoxelMap& map, uint64_t index) { return map.download_occupancy(index, 1).front(); }
}

TEST_F(VulkanTest, ProbMapStartsUnknownAndReportsGeometry)
{
	ProbVoxelMap map(ctx(), kProbDims, kSide);
	EXPECT_EQ(map.voxel_count(), 23u * 19u * 17u);
	EXPECT_EQ(map.memory_usage_bytes(), 23u * 19u * 17u * 4u);
	EXPECT_TRUE(equal_to_reference(map, prob_reference()));
	EXPECT_EQ(value_at(map, 0), -128);
	EXPECT_TRUE(map.download_occupancy(map.voxel_count(), 5).empty());
	EXPECT_THROW((void)map.download_occupancy(map.voxel_count() + 1), VulkanError);
}

TEST_F(VulkanTest, ProbMapRejectsInvalidConstruction)
{
	EXPECT_THROW(ProbVoxelMap(ctx(), { 0, 4, 4 }, 1.0f), VulkanError);
	EXPECT_THROW(ProbVoxelMap(ctx(), { 4, 4, 4 }, 0.0f), VulkanError);
	EXPECT_THROW(ProbVoxelMap(ctx(), { 4, 4, 4 }, std::nanf("")), VulkanError);
	EXPECT_THROW(ProbVoxelMap(ctx(), { (1u << 24) + 1, 1, 1 }, 1.0f), VulkanError);
	EXPECT_THROW(ProbVoxelMap(ctx(), { 4096, 4096, 4096 }, 1.0f), VulkanError);
}

TEST_F(VulkanTest, ProbabilityConversionsFollowCuda)
{
	EXPECT_EQ(ProbVoxelMap::probability_from_float(0.0f), -127);
	EXPECT_EQ(ProbVoxelMap::probability_from_float(1.0f), 127);
	EXPECT_EQ(ProbVoxelMap::probability_from_float(0.5f), 0);
	EXPECT_EQ(ProbVoxelMap::probability_from_float(0.25f), -63);   // -127 + 63.5 truncated towards zero
	EXPECT_EQ(ProbVoxelMap::probability_from_float(0.75f), 63);    // -127 + 190.5 = 63.5
	EXPECT_EQ(ProbVoxelMap::probability_from_float(-3.0f), -127);  // clamped
	EXPECT_EQ(ProbVoxelMap::probability_from_float(7.0f), 127);
	EXPECT_EQ(ProbVoxelMap::probability_from_float(std::nanf("")), 127); // std::min(1, NaN) is 1
	for (float p : { 0.0f, 0.1f, 0.5f, 0.9f, 1.0f })
		EXPECT_NEAR(ProbVoxelMap::probability_to_float(ProbVoxelMap::probability_from_float(p)), p, 1.0f / 254.0f);
	EXPECT_FLOAT_EQ(ProbVoxelMap::probability_to_float(-127), 0.0f);
	EXPECT_FLOAT_EQ(ProbVoxelMap::probability_to_float(127), 1.0f);
	for (float p = -0.5f; p < 1.5f; p += 0.013f)
		EXPECT_EQ(ProbVoxelMap::probability_from_float(p), ProbVoxelMapReference::probability_from_float(p)) << p;
}

TEST_F(VulkanTest, ProbMapMeaningsUpdateLikeCuda)
{
	// voxel (1,1,1) of a 4x4x4 map, side 1: index (1*4+1)*4+1 = 21
	const Point3f point{ 1.5f, 1.5f, 1.5f };
	const std::vector<Point3f> points{ point };
	struct Case { uint8_t meaning; int expected_from_unknown; };
	const Case cases[] = { { eBVM_FREE, -10 }, { eBVM_OCCUPIED, 72 }, { eBVM_COLLISION, 72 }, { eBVM_UNKNOWN, -128 },
		{ eBVM_UNDEFINED, -128 }, { 4, -125 }, { 128, -1 }, { 129, 0 }, { 130, 1 }, { 200, 71 }, { 254, 125 } };
	for (UpdateMode mode : kAllModes)
		for (const Case& c : cases)
		{
			ProbVoxelMap map(ctx(), { 4, 4, 4 }, 1.0f);
			EXPECT_EQ(map.insert_point_cloud(points, c.meaning, with_mode(mode)), 0u);
			EXPECT_EQ(value_at(map, 21), c.expected_from_unknown) << mode_name(mode) << " meaning " << int(c.meaning);
			EXPECT_EQ(value_at(map, 20), -128) << "neighbours stay unknown";
			EXPECT_EQ(value_at(map, 22), -128);
		}

	for (UpdateMode mode : kAllModes)
	{
		ProbVoxelMap map(ctx(), { 4, 4, 4 }, 1.0f);
		map.insert_point_cloud(points, eBVM_OCCUPIED, with_mode(mode));
		map.insert_point_cloud(points, eBVM_OCCUPIED, with_mode(mode));
		EXPECT_EQ(value_at(map, 21), 127) << mode_name(mode) << ": 72 + 72 saturates";
		map.insert_point_cloud(points, eBVM_FREE, with_mode(mode));
		EXPECT_EQ(value_at(map, 21), 117) << mode_name(mode);
		map.insert_point_cloud(points, eBVM_UNKNOWN, with_mode(mode));
		EXPECT_EQ(value_at(map, 21), -128) << mode_name(mode) << ": UNKNOWN resets";
		map.insert_point_cloud(points, 4, with_mode(mode));
		for (int i = 0; i < 3; ++i)
			map.insert_point_cloud(points, 4, with_mode(mode));
		EXPECT_EQ(value_at(map, 21), -127) << mode_name(mode) << ": -125 four times saturates at -127";
	}

	// a custom sensor model changes FREE / OCCUPIED / COLLISION but not the swept volume meanings
	const SensorModel model{ -3, 5 };
	for (UpdateMode mode : kAllModes)
	{
		ProbVoxelMap map(ctx(), { 4, 4, 4 }, 1.0f);
		map.insert_point_cloud(points, eBVM_OCCUPIED, with_mode(mode, model));
		EXPECT_EQ(value_at(map, 21), 5);
		map.insert_point_cloud(points, eBVM_FREE, with_mode(mode, model));
		EXPECT_EQ(value_at(map, 21), 2);
		map.insert_point_cloud(points, 130, with_mode(mode, model));
		EXPECT_EQ(value_at(map, 21), 3);
	}
}

TEST_F(VulkanTest, ProbMapRejectsInvalidSensorModelAndMeaningOperations)
{
	ProbVoxelMap map(ctx(), { 4, 4, 4 }, 1.0f);
	const std::vector<Point3f> points{ { 1.5f, 1.5f, 1.5f } };
	EXPECT_THROW(map.insert_point_cloud(points, eBVM_OCCUPIED, with_mode(UpdateMode::Atomic, SensorModel{ -128, 72 })), VulkanError);
	EXPECT_THROW(map.insert_point_cloud(points, eBVM_OCCUPIED, with_mode(UpdateMode::Atomic, SensorModel{ -10, -128 })), VulkanError);
	EXPECT_THROW((void)map.insert_sensor_data(points, { 0.5f, 0.5f, 0.5f }, true, nullptr, eBVM_OCCUPIED, with_mode(UpdateMode::Atomic, SensorModel{ -128, 1 })), VulkanError);
	EXPECT_EQ(value_at(map, 21), -128) << "rejected calls must not change the map";

	map.insert_point_cloud(points, eBVM_OCCUPIED);
	EXPECT_THROW(map.clear_bit_voxel_meaning(eBVM_FREE), VulkanError);
	EXPECT_THROW(map.clear_bit_voxel_meaning(7), VulkanError);
	EXPECT_EQ(value_at(map, 21), 72);
	map.clear_bit_voxel_meaning(eBVM_OCCUPIED); // CUDA: clears the whole map
	EXPECT_EQ(value_at(map, 21), -128);
	map.insert_point_cloud(points, eBVM_OCCUPIED);
	map.clear_map();
	EXPECT_TRUE(equal_to_reference(map, prob_reference({ 4, 4, 4 }, 1.0f)));
}

TEST_F(VulkanTest, ProbMapSequentialModeIsExactEvenWithSaturation)
{
	const Dim3ui dims{ 6, 6, 6 }; // few voxels: many updates per voxel, both saturation ends are reached
	ProbVoxelMap map(ctx(), dims, kSide);
	auto ref = prob_reference(dims);
	std::mt19937 rng(61);
	const uint8_t meanings[] = { eBVM_OCCUPIED, eBVM_FREE, eBVM_FREE, eBVM_OCCUPIED, 100, 200, eBVM_FREE };
	for (uint8_t meaning : meanings)
	{
		const auto points = random_points(rng, 3000, dims);
		const int increment = meaning == eBVM_FREE ? -10 : (meaning == eBVM_OCCUPIED ? 72 : meaning - 129);
		const auto generated = ref.point_events(points, increment);
		EXPECT_EQ(map.insert_point_cloud(points, meaning, with_mode(UpdateMode::Sequential)), generated.outside);
		ref.apply_sequential(generated.events);
		ASSERT_TRUE(equal_to_reference(map, ref)) << "meaning " << int(meaning);
	}
	bool saw_max = false, saw_min = false;
	for (int8_t v : map.download_occupancy())
	{
		saw_max |= v == 127;
		saw_min |= v == -127;
	}
	EXPECT_TRUE(saw_max || saw_min) << "the test must reach saturation";
}

TEST_F(VulkanTest, ProbMapAccumulateModeAppliesPerVoxelSumsAndResetsBetweenCalls)
{
	const Dim3ui dims{ 6, 6, 6 };
	ProbVoxelMap map(ctx(), dims, kSide);
	auto ref = prob_reference(dims);
	std::mt19937 rng(62);
	const uint8_t meanings[] = { eBVM_OCCUPIED, eBVM_FREE, eBVM_OCCUPIED, 110, eBVM_FREE, eBVM_UNKNOWN, eBVM_OCCUPIED };
	for (uint8_t meaning : meanings)
	{
		const auto points = random_points(rng, 2000, dims);
		if (meaning == eBVM_UNKNOWN)
		{
			ref.assign_unknown(ref.point_events(points, 0).events);
			map.insert_point_cloud(points, meaning, with_mode(UpdateMode::Accumulate));
		}
		else
		{
			const int increment = meaning == eBVM_FREE ? -10 : (meaning == eBVM_OCCUPIED ? 72 : meaning - 129);
			ref.apply_accumulated(ref.point_events(points, increment).events);
			map.insert_point_cloud(points, meaning, with_mode(UpdateMode::Accumulate));
		}
		ASSERT_TRUE(equal_to_reference(map, ref)) << "meaning " << int(meaning);
	}
}

TEST_F(VulkanTest, ProbMapAccumulateEqualsSequentialWhenNothingSaturates)
{
	std::mt19937 rng(63);
	const auto points = random_points(rng, 400, kProbDims); // a few hits per voxel at most: no saturation
	ProbVoxelMap sequential(ctx(), kProbDims, kSide), accumulate(ctx(), kProbDims, kSide), atomic(ctx(), kProbDims, kSide);
	sequential.insert_point_cloud(points, eBVM_FREE, with_mode(UpdateMode::Sequential));
	accumulate.insert_point_cloud(points, eBVM_FREE, with_mode(UpdateMode::Accumulate));
	atomic.insert_point_cloud(points, eBVM_FREE, with_mode(UpdateMode::Atomic));
	EXPECT_EQ(accumulate.download_occupancy(), sequential.download_occupancy());
	EXPECT_EQ(atomic.download_occupancy(), sequential.download_occupancy());
	auto ref = prob_reference();
	ref.apply_sequential(ref.point_events(points, -10).events);
	EXPECT_TRUE(equal_to_reference(sequential, ref));
}

TEST_F(VulkanTest, ProbMapUpdatesAreNeverLostUnderContention)
{
	// 200 voxels, 100 simultaneous +1 updates each (never saturates): every mode must deliver the exact sum
	const Dim3ui dims{ 200, 1, 1 };
	std::vector<Coord3ui> all;
	for (uint32_t x = 0; x < dims.x; ++x)
		all.push_back({ x, 0, 0 });
	std::vector<Point3f> points;
	for (uint32_t i = 0; i < 20000; ++i)
		points.push_back({ static_cast<float>(i / 100) + 0.5f, 0.5f, 0.5f }); // neighbouring threads hit the same voxel
	for (UpdateMode mode : kAllModes)
	{
		ProbVoxelMap map(ctx(), dims, 1.0f);
		map.insert_coordinate_list(all, 129, with_mode(mode)); // every voxel becomes known with occupancy 0
		EXPECT_EQ(map.insert_point_cloud(points, 130, with_mode(mode)), 0u); // +1 each
		for (int8_t v : map.download_occupancy())
			ASSERT_EQ(v, 100) << mode_name(mode);
	}
}

TEST_F(VulkanTest, ProbMapAtomicModeResultIsASerialOrderOfTheUpdates)
{
	// Voxel 0 sits at 120; three measurements ray from voxel 3 to voxel 0: each updates voxel 0 with -10 (free space) and
	// then +72 (hit). Saturation makes the result depend on the order; every result must be that of SOME serial order.
	const std::set<int> allowed = ProbVoxelMapReference::serializations(120, { -10, 72, -10, 72, -10, 72 });
	ASSERT_GT(allowed.size(), 1u) << "the scenario must be order dependent";
	const std::vector<Point3f> hits(3, Point3f{ 0.5f, 0.5f, 0.5f });
	for (int repetition = 0; repetition < 40; ++repetition)
	{
		ProbVoxelMap map(ctx(), { 4, 1, 1 }, 1.0f);
		map.insert_point_cloud(std::vector<Point3f>{ { 0.5f, 0.5f, 0.5f } }, 249, with_mode(UpdateMode::Atomic)); // +120
		map.insert_sensor_data(hits, { 3.5f, 0.5f, 0.5f }, true, nullptr, eBVM_OCCUPIED, with_mode(UpdateMode::Atomic));
		const int result = value_at(map, 0);
		EXPECT_EQ(allowed.count(result), 1u) << "value " << result << " is not the result of any serial order";
		EXPECT_EQ(value_at(map, 1), -30) << "unknown voxels on the ray: three free updates";
	}
}

TEST_F(VulkanTest, ProbMapInsertsCoordinateListsInAllModes)
{
	std::mt19937 rng(64);
	std::uniform_int_distribution<uint32_t> vx(0, kProbDims.x + 2), vy(0, kProbDims.y + 2), vz(0, kProbDims.z + 2);
	std::vector<Coord3ui> coordinates;
	for (int i = 0; i < 3000; ++i)
		coordinates.push_back({ vx(rng) % 6, vy(rng) % 6, vz(rng) % 6 });
	coordinates.push_back({ kProbDims.x, 0, 0 });
	coordinates.push_back({ 0, kProbDims.y, 0 });
	coordinates.push_back({ 0, 0, kProbDims.z });
	coordinates.push_back({ UINT32_MAX, UINT32_MAX, 0 });
	for (UpdateMode mode : { UpdateMode::Accumulate, UpdateMode::Sequential })
	{
		ProbVoxelMap map(ctx(), kProbDims, kSide);
		auto ref = prob_reference();
		const auto generated = ref.coordinate_events(coordinates, 72);
		EXPECT_EQ(generated.outside, 4u);
		EXPECT_EQ(map.insert_coordinate_list(coordinates, eBVM_OCCUPIED, with_mode(mode)), 4u);
		apply_to_reference(ref, generated.events, mode);
		EXPECT_TRUE(equal_to_reference(map, ref)) << mode_name(mode);
	}
}

TEST_F(VulkanTest, ProbMapDropsPointsOutsideTheMap)
{
	ProbVoxelMap map(ctx(), kProbDims, kSide);
	const float nan = std::numeric_limits<float>::quiet_NaN();
	const float inf = std::numeric_limits<float>::infinity();
	const std::vector<Point3f> points{ { 0.55f, 0.55f, 0.55f }, { -0.05f, 0.5f, 0.5f }, { 0.5f, -0.05f, 0.5f }, { 0.5f, 0.5f, -0.05f },
		{ 2.35f, 0.5f, 0.5f }, { 0.5f, 1.95f, 0.5f }, { 0.5f, 0.5f, 1.75f }, { nan, 0.5f, 0.5f }, { 0.5f, nan, 0.5f },
		{ 0.5f, 0.5f, nan }, { inf, 0.5f, 0.5f }, { 0.5f, -inf, 0.5f }, { 2.25f, 1.85f, 1.65f } };
	for (UpdateMode mode : kAllModes)
	{
		map.clear_map();
		EXPECT_EQ(map.insert_point_cloud(points, eBVM_OCCUPIED, with_mode(mode)), 11u) << mode_name(mode);
		auto ref = prob_reference();
		ref.apply_sequential(ref.point_events(points, 72).events);
		EXPECT_TRUE(equal_to_reference(map, ref)) << mode_name(mode);
	}
}

TEST_F(VulkanTest, ProbMapMetaCloudsApplySubCloudsInOrder)
{
	std::mt19937 rng(65);
	const Dim3ui dims{ 8, 8, 8 };
	const std::vector<std::vector<Point3f>> clouds{ random_points(rng, 800, dims), {}, random_points(rng, 700, dims), random_points(rng, 600, dims),
		random_points(rng, 500, dims) };
	MetaPointCloud meta(ctx(), clouds);
	// sub clouds 0..2 share one update (the implementation applies such neighbours as ONE batch; since all their increments
	// have the same sign this is only a speed-up, not observable in the result), 3 resets the voxels it hits to unknown,
	// 4 is a swept volume meaning (-9)
	const std::vector<uint8_t> meanings{ eBVM_OCCUPIED, eBVM_OCCUPIED, eBVM_OCCUPIED, eBVM_UNKNOWN, 120 };
	for (UpdateMode mode : { UpdateMode::Accumulate, UpdateMode::Sequential })
	{
		ProbVoxelMap map(ctx(), dims, kSide);
		auto ref = prob_reference(dims);
		std::vector<ProbVoxelMapReference::Event> occupied_run;
		uint32_t expected_outside = 0;
		for (std::size_t k = 0; k < 3; ++k)
		{
			const auto g = ref.point_events(clouds[k], 72);
			expected_outside += g.outside;
			occupied_run.insert(occupied_run.end(), g.events.begin(), g.events.end());
		}
		apply_to_reference(ref, occupied_run, mode);
		const auto unknown = ref.point_events(clouds[3], 0);
		ref.assign_unknown(unknown.events);
		const auto swept = ref.point_events(clouds[4], 120 - 129);
		apply_to_reference(ref, swept.events, mode);
		expected_outside += unknown.outside + swept.outside;

		EXPECT_EQ(map.insert_meta_point_cloud(meta, meanings, with_mode(mode)), expected_outside) << mode_name(mode);
		EXPECT_TRUE(equal_to_reference(map, ref)) << mode_name(mode);
	}

	ProbVoxelMap map(ctx(), dims, kSide);
	auto ref = prob_reference(dims);
	EXPECT_EQ(map.insert_meta_point_cloud(meta, eBVM_FREE, with_mode(UpdateMode::Sequential)), 0u);
	for (const auto& cloud : clouds)
		ref.apply_sequential(ref.point_events(cloud, -10).events);
	EXPECT_TRUE(equal_to_reference(map, ref));
	EXPECT_THROW(map.insert_meta_point_cloud(meta, std::vector<uint8_t>{ 1, 2 }), VulkanError);
}

TEST_F(VulkanTest, ProbMapCollidesWithBitAndProbMaps)
{
	std::mt19937 rng(66);
	const Dim3ui dims = kProbDims;
	auto build = [&](ProbVoxelMap& map, ProbVoxelMapReference& ref, uint32_t seed)
	{
		std::mt19937 local(seed);
		for (uint8_t meaning : { uint8_t{ eBVM_OCCUPIED }, uint8_t{ eBVM_FREE }, uint8_t{ 150 }, uint8_t{ eBVM_OCCUPIED }, uint8_t{ 60 } })
		{
			const auto points = random_points(local, 1500, dims);
			const int inc = meaning == eBVM_FREE ? -10 : (meaning == eBVM_OCCUPIED ? 72 : meaning - 129);
			map.insert_point_cloud(points, meaning, with_mode(UpdateMode::Sequential));
			ref.apply_sequential(ref.point_events(points, inc).events);
		}
	};
	BitVoxelMap bits(ctx(), dims, kSide);
	auto bits_ref = map_reference(dims);
	for (uint8_t meaning : { uint8_t{ eBVM_OCCUPIED }, uint8_t{ eBVM_FREE }, uint8_t{ 77 } })
	{
		const auto points = random_points(rng, 1500, dims);
		bits.insert_point_cloud(points, meaning);
		bits_ref.insert_points(points, meaning);
	}
	uint32_t distinct_counts = 0;
	uint32_t previous = UINT32_MAX;
	for (float threshold : { 0.0f, 0.25f, 0.5f, 0.75f, 0.9f, 1.0f, -0.5f, 2.0f, std::nanf("") })
	{
		ProbVoxelMap map(ctx(), dims, kSide);
		auto ref = prob_reference(dims);
		build(map, ref, 7);
		const uint32_t expected = ref.collide_with(bits_ref, threshold, 72);
		EXPECT_EQ(map.collide_with(bits, threshold), expected) << "threshold " << threshold;
		EXPECT_TRUE(equal_to_reference(map, ref)) << "collision marking, threshold " << threshold;
		if (expected != previous)
			++distinct_counts;
		previous = expected;
	}
	EXPECT_GE(distinct_counts, 3u) << "the thresholds must matter in this test";

	for (float threshold : { 0.0f, 0.5f, 0.9f, 1.0f })
	{
		ProbVoxelMap a(ctx(), dims, kSide), b(ctx(), dims, kSide);
		auto ref_a = prob_reference(dims), ref_b = prob_reference(dims);
		build(a, ref_a, 8);
		build(b, ref_b, 9);
		const uint32_t expected = ref_a.collide_with(ref_b, threshold, 72);
		EXPECT_EQ(a.collide_with(b, threshold), expected) << threshold;
		EXPECT_TRUE(equal_to_reference(a, ref_a));
		EXPECT_TRUE(equal_to_reference(b, ref_b)) << "the other map is not modified";
	}

	// a custom marker increment
	{
		ProbVoxelMap map(ctx(), dims, kSide);
		auto ref = prob_reference(dims);
		build(map, ref, 7);
		const uint32_t expected = ref.collide_with(bits_ref, 0.5f, 5);
		EXPECT_EQ(map.collide_with(bits, 0.5f, SensorModel{ -10, 5 }), expected);
		EXPECT_TRUE(equal_to_reference(map, ref));
	}
}

TEST_F(VulkanTest, BitMapCollidesWithProbMapAndMarksItsOwnVoxels)
{
	std::mt19937 rng(67);
	const Dim3ui dims = kProbDims;
	ProbVoxelMap prob(ctx(), dims, kSide);
	auto prob_ref = prob_reference(dims);
	for (uint8_t meaning : { uint8_t{ eBVM_OCCUPIED }, uint8_t{ eBVM_FREE }, uint8_t{ 150 } })
	{
		const auto points = random_points(rng, 1500, dims);
		const int inc = meaning == eBVM_FREE ? -10 : (meaning == eBVM_OCCUPIED ? 72 : meaning - 129);
		prob.insert_point_cloud(points, meaning, with_mode(UpdateMode::Sequential));
		prob_ref.apply_sequential(prob_ref.point_events(points, inc).events);
	}
	for (float threshold : { 0.0f, 0.5f, 0.75f, 1.0f, 3.0f })
	{
		BitVoxelMap bits(ctx(), dims, kSide);
		auto bits_ref = map_reference(dims);
		for (uint8_t meaning : { uint8_t{ eBVM_OCCUPIED }, uint8_t{ eBVM_FREE }, uint8_t{ 9 } })
		{
			std::mt19937 local(100 + meaning);
			const auto points = random_points(local, 1500, dims);
			bits.insert_point_cloud(points, meaning);
			bits_ref.insert_points(points, meaning);
		}
		const uint32_t expected = ProbVoxelMapReference::collide_bit_with_prob(bits_ref, prob_ref, threshold);
		EXPECT_EQ(bits.collide_with(prob, threshold), expected) << threshold;
		EXPECT_EQ(bits.download_words(), bits_ref.to_words(0, bits.voxel_count())) << threshold;
	}
	BitVoxelMap wrong(ctx(), { 5, 5, 5 }, kSide);
	EXPECT_THROW((void)wrong.collide_with(prob), VulkanError);
}

TEST_F(VulkanTest, ProbMapCollisionRejectsMismatchingMaps)
{
	ProbVoxelMap a(ctx(), { 4, 4, 4 }, 1.0f), b(ctx(), { 4, 4, 5 }, 1.0f);
	BitVoxelMap bit(ctx(), { 4, 4, 5 }, 1.0f);
	EXPECT_THROW((void)a.collide_with(b), VulkanError);
	EXPECT_THROW((void)a.collide_with(bit), VulkanError);
	EXPECT_THROW((void)a.collide_with(a), VulkanError);
	EXPECT_THROW((void)a.collide_with(ProbVoxelMap(ctx(), { 4, 4, 4 }, 1.0f), 0.5f, SensorModel{ -128, 0 }), VulkanError);
}
