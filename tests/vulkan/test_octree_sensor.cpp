// Slice S6f-1: Octree::voxelize_points, the first step of the sensor pipeline (pose, voxel grid of the pipeline, distinct voxels with the number of points
// and the occupancy of the sensor model). Compared with a plain host reference that works in double precision (std::map of voxel -> count); the test
// points are built from a chosen world position inside a voxel (away from its border) and the inverse pose, so float rounding on the GPU cannot change
// a voxel. Border cases (points exactly on voxel borders, negative zero) use poses with entries 0 / +-1 and side lengths that are powers of two, where
// float arithmetic is exact.
#include "vulkan_fixture.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>

#include "../reference/octree_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference::octree;

namespace
{
	constexpr OctreeKind kKinds[] = { OctreeKind::Deterministic, OctreeKind::Probabilistic };

	/// Rotation about x, y, z (radians, applied in that order) and a translation.
	Matrix4f make_pose(float rx, float ry, float rz, float tx, float ty, float tz)
	{
		Matrix4f x, y, z;
		x(1, 1) = std::cos(rx); x(1, 2) = -std::sin(rx); x(2, 1) = std::sin(rx); x(2, 2) = std::cos(rx);
		y(0, 0) = std::cos(ry); y(0, 2) = std::sin(ry); y(2, 0) = -std::sin(ry); y(2, 2) = std::cos(ry);
		z(0, 0) = std::cos(rz); z(0, 1) = -std::sin(rz); z(1, 0) = std::sin(rz); z(1, 1) = std::cos(rz);
		Matrix4f m = z * y * x;
		m(0, 3) = tx; m(1, 3) = ty; m(2, 3) = tz;
		return m;
	}

	/// The sensor point that the pose maps to `world` (inverse of R * p + t, with the stored float entries).
	Point3f sensor_point(const Matrix4f& pose, double wx, double wy, double wz)
	{
		const double dx = wx - pose(0, 3), dy = wy - pose(1, 3), dz = wz - pose(2, 3);
		return { float(pose(0, 0) * dx + pose(1, 0) * dy + pose(2, 0) * dz), float(pose(0, 1) * dx + pose(1, 1) * dy + pose(2, 1) * dz),
			float(pose(0, 2) * dx + pose(1, 2) * dy + pose(2, 2) * dz) };
	}

	struct Expected
	{
		std::vector<OctreeVoxelRun> runs;
		uint32_t outside = 0;
	};

	/// The reference: double arithmetic, one map entry per voxel, occupancy from the unclamped count.
	Expected reference(const std::vector<Point3f>& points, const Matrix4f& pose, float side, const OctreeSensorModel& model)
	{
		std::map<uint64_t, uint32_t> counts;
		Expected e;
		for (const Point3f& p : points)
		{
			double w[3];
			for (int r = 0; r < 3; ++r)
				w[r] = double(pose(r, 0)) * p.x + double(pose(r, 1)) * p.y + double(pose(r, 2)) * p.z + pose(r, 3);
			const double fx = std::floor(w[0] / side), fy = std::floor(w[1] / side), fz = std::floor(w[2] / side);
			if (!(fx >= 0 && fx < 16384 && fy >= 0 && fy < 16384 && fz >= 0 && fz < 16384))
			{
				++e.outside;
				continue;
			}
			++counts[ref::morton(uint32_t(fx), uint32_t(fy), uint32_t(fz))];
		}
		for (const auto& [id, count] : counts)
		{
			const int64_t v = int64_t(model.initial_probability) + int64_t(model.update_probability) * count;
			e.runs.push_back({ id, count, static_cast<int8_t>(std::clamp<int64_t>(v, -127, 127)) });
		}
		return e;
	}

	/// `count` random points: world position inside random voxels (0 .. extent - 1), away from the borders, about a third of them repeated.
	std::vector<Point3f> random_points(std::mt19937_64& rng, const Matrix4f& pose, float side, size_t count, uint32_t extent)
	{
		std::vector<Point3f> points;
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::uniform_real_distribution<double> f(0.25, 0.75);
		for (size_t i = 0; i < count; ++i)
		{
			if (i % 3 == 2 && !points.empty())
				points.push_back(points[rng() % points.size()]);
			else
				points.push_back(sensor_point(pose, (d(rng) + f(rng)) * side, (d(rng) + f(rng)) * side, (d(rng) + f(rng)) * side));
		}
		return points;
	}

	void expect_same(Octree& tree, const std::vector<Point3f>& points, const Matrix4f& pose, float side, const OctreeSensorModel& model, const char* what)
	{
		const Expected e = reference(points, pose, side, model);
		uint32_t outside = 12345;
		const auto got = tree.voxelize_points(points, pose, side, model, &outside);
		EXPECT_EQ(got.size(), e.runs.size()) << what;
		for (size_t i = 0; i < std::min(got.size(), e.runs.size()); ++i)
			if (!(got[i] == e.runs[i]))
			{
				ADD_FAILURE() << what << ": run " << i << " is (id " << got[i].id << ", count " << got[i].count << ", occupancy " << int(got[i].occupancy) << ") but the reference has (id "
					<< e.runs[i].id << ", count " << e.runs[i].count << ", occupancy " << int(e.runs[i].occupancy) << ")";
				break;
			}
		EXPECT_EQ(outside, e.outside) << what;
	}
}

TEST_F(VulkanTest, OctreeVoxelizeMatchesReference)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(11);
	for (const OctreeKind kind : kKinds)
	{
		Octree tree(ctx(), kind);
		const Matrix4f poses[] = { Matrix4f::identity(), make_pose(0.3f, -0.7f, 1.9f, 3.25f, -1.0f, 0.5f), make_pose(3.0f, 0.2f, -2.5f, 40.0f, 25.5f, 31.0f) };
		const OctreeSensorModel models[] = { {}, { 0, 100 }, { 0, -10 }, { 5, -3 }, { 127, 127 }, { -128, -128 } };
		for (const Matrix4f& pose : poses)
			for (const float side : { 0.05f, 0.125f, 0.5f })
				for (const size_t n : { size_t(1), size_t(2), size_t(7), size_t(1000), size_t(30000) })
				{
					const auto points = random_points(rng, pose, side, n, 150);
					expect_same(tree, points, pose, side, models[rng() % std::size(models)], "random cloud");
				}
		// the whole range of models on one cloud
		const Matrix4f pose = make_pose(-1.1f, 0.4f, 0.8f, 20.0f, 20.0f, 20.0f);
		const auto points = random_points(rng, pose, 0.25f, 5000, 40);
		for (const OctreeSensorModel& m : models)
			expect_same(tree, points, pose, 0.25f, m, "models");
	}
}

TEST_F(VulkanTest, OctreeVoxelizeCounts)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	const Matrix4f pose = Matrix4f::identity();
	// k points in one voxel next to k' points in another: the counts around the clamp of the sensor model (the shader limits the count to 255)
	for (const uint32_t k : { 1u, 2u, 3u, 126u, 127u, 128u, 254u, 255u, 256u, 257u, 1000u, 70000u })
	{
		std::vector<Point3f> points(k, Point3f{ 0.55f, 0.55f, 0.55f });
		points.push_back({ 5.5f, 0.5f, 0.5f });
		points.push_back({ 5.5f, 0.5f, 0.5f });
		for (const OctreeSensorModel& m : { OctreeSensorModel{ 0, 1 }, OctreeSensorModel{ 0, -1 }, OctreeSensorModel{ 0, 100 }, OctreeSensorModel{ 100, 1 }, OctreeSensorModel{ -100, -1 },
			OctreeSensorModel{ 0, 0 }, OctreeSensorModel{ 127, -1 }, OctreeSensorModel{ -128, 1 } })
			expect_same(tree, points, pose, 1.0f, m, "count");
	}
	// a plain check of the numbers, not through the reference: 3 points, update 100 -> clamp 127; update -10 -> -30
	const std::vector<Point3f> three(3, Point3f{ 2.5f, 3.5f, 4.5f });
	const auto a = tree.voxelize_points(three, pose, 1.0f, { 0, 100 });
	ASSERT_EQ(a.size(), 1u);
	EXPECT_EQ(a[0].id, ref::morton(2, 3, 4));
	EXPECT_EQ(a[0].count, 3u);
	EXPECT_EQ(a[0].occupancy, 127);
	EXPECT_EQ(tree.voxelize_points(three, pose, 1.0f, { 0, -10 })[0].occupancy, -30);
	EXPECT_EQ(tree.voxelize_points(three, pose, 1.0f, { -100, -10 })[0].occupancy, -127);
	EXPECT_EQ(tree.voxelize_points(three, pose, 1.0f, { 7, 0 })[0].occupancy, 7);
}

TEST_F(VulkanTest, OctreeVoxelizeBorders)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Probabilistic);
	// side 0.25 and poses with entries 0 / +-1: every product and sum is exact, so the points sit exactly on voxel borders
	Matrix4f permute; // x' = -z + 2, y' = x + 0.5, z' = y
	permute.m = { 0, 0, -1, 2.0f, 1, 0, 0, 0.5f, 0, 1, 0, 0, 0, 0, 0, 1 };
	for (const Matrix4f& pose : { Matrix4f::identity(), permute })
	{
		std::vector<Point3f> points;
		for (int i = -4; i <= 12; ++i)
			points.push_back({ 0.25f * float(i), 0.25f * float(i / 2), 0.25f * float(12 - i) });
		points.push_back({ -0.0f, -0.0f, -0.0f });
		points.push_back({ 0.0f, 0.0f, 0.0f });
		if (pose == Matrix4f::identity()) // with a translation the tiny values vanish in float arithmetic (0.5 - 1e-9 = 0.5), but not in the double reference
		{
			points.push_back({ -1e-9f, 1.0f, 1.0f });
			points.push_back({ 1.0f, -1e-9f, 1.0f });
			points.push_back({ 1.0f, 1.0f, -1e-9f });
		}
		points.push_back({ 4095.75f, 0.0f, 0.0f });
		points.push_back({ 4096.0f, 0.0f, 0.0f }); // voxel 16384: outside
		points.push_back({ 4095.999f, 4095.75f, 4095.75f }); // 4095.999 / 0.25 = 16383.99 -> still inside
		expect_same(tree, points, pose, 0.25f, { 0, 100 }, "borders");
	}
	// the last voxel of the grid keeps the largest id, 2^42 - 1
	const auto last = tree.voxelize_points(std::vector<Point3f>{ { 16383.5f, 16383.5f, 16383.5f } }, Matrix4f::identity(), 1.0f);
	ASSERT_EQ(last.size(), 1u);
	EXPECT_EQ(last[0].id, (uint64_t(1) << 42) - 1);
	// negative zero belongs to voxel 0, a tiny negative value does not
	uint32_t outside = 0;
	const auto zero = tree.voxelize_points(std::vector<Point3f>{ { -0.0f, -0.0f, -0.0f } }, Matrix4f::identity(), 1.0f, {}, &outside);
	ASSERT_EQ(zero.size(), 1u);
	EXPECT_EQ(zero[0].id, 0u);
	EXPECT_EQ(outside, 0u);
	const auto tiny = tree.voxelize_points(std::vector<Point3f>{ { -1e-9f, 0.5f, 0.5f } }, Matrix4f::identity(), 1.0f, {}, &outside);
	EXPECT_TRUE(tiny.empty());
	EXPECT_EQ(outside, 1u);
}

TEST_F(VulkanTest, OctreeVoxelizeOutsidePointsAndPoses)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
	const Matrix4f id = Matrix4f::identity();
	std::mt19937_64 rng(5);
	// bad points between good ones, every kind of bad
	std::vector<Point3f> points = random_points(rng, id, 0.5f, 500, 50);
	const std::vector<Point3f> bad = { { nan, 1, 1 }, { 1, nan, 1 }, { 1, 1, nan }, { inf, 1, 1 }, { 1, -inf, 1 }, { 1, 1, inf }, { -1, 1, 1 }, { 1, -1, 1 }, { 1, 1, -1 },
		{ 8192.0f, 1, 1 }, { 1, 8192.0f, 1 }, { 1, 1, 8192.0f }, { 1e30f, 1, 1 }, { -1e30f, 1, 1 } };
	points.insert(points.begin() + 100, bad.begin(), bad.end());
	expect_same(tree, points, id, 0.5f, { 0, 50 }, "bad points");
	{
		uint32_t outside = 0;
		const auto r = tree.voxelize_points(bad, id, 0.5f, {}, &outside);
		EXPECT_TRUE(r.empty());
		EXPECT_EQ(outside, bad.size());
	}
	// a pose that moves everything below zero, a pose with a NaN entry, a pose that scales by 0
	Matrix4f below = id;
	below(0, 3) = -1000.0f;
	expect_same(tree, points, below, 0.5f, {}, "pose below zero");
	Matrix4f poisoned = id;
	poisoned(1, 1) = nan;
	{
		uint32_t outside = 0;
		EXPECT_TRUE(tree.voxelize_points(points, poisoned, 0.5f, {}, &outside).empty());
		EXPECT_EQ(outside, points.size());
	}
	Matrix4f flat = id;
	flat(2, 2) = 0.0f;
	flat(2, 3) = 1.25f;
	expect_same(tree, points, flat, 0.5f, {}, "flat pose");
	// the last row of the pose is ignored
	Matrix4f odd_row = make_pose(0.5f, 0.2f, 0.1f, 10, 10, 10);
	const auto pts = random_points(rng, odd_row, 0.25f, 300, 60);
	const auto before = tree.voxelize_points(pts, odd_row, 0.25f);
	odd_row(3, 0) = 7.0f;
	odd_row(3, 3) = 0.0f;
	EXPECT_EQ(tree.voxelize_points(pts, odd_row, 0.25f), before);
}

TEST_F(VulkanTest, OctreeVoxelizeEdgeCases)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	const Matrix4f id = Matrix4f::identity();
	uint32_t outside = 99;
	EXPECT_TRUE(tree.voxelize_points({}, id, 1.0f, {}, &outside).empty());
	EXPECT_EQ(outside, 0u);
	// side lengths that are not positive and finite throw, also for an empty cloud
	const float nan = std::numeric_limits<float>::quiet_NaN();
	for (const float side : { 0.0f, -1.0f, nan, std::numeric_limits<float>::infinity() })
	{
		EXPECT_THROW((void)tree.voxelize_points({}, id, side), VulkanError) << side;
		EXPECT_THROW((void)tree.voxelize_points(std::vector<Point3f>{ { 1, 1, 1 } }, id, side), VulkanError) << side;
	}
	// one point, the side length of the tree is not used and the tree does not change
	Octree sized(ctx(), OctreeKind::Probabilistic, 0.5f);
	const auto before = sized.extract();
	const auto one = sized.voxelize_points(std::vector<Point3f>{ { 3.0f, 3.0f, 3.0f } }, id, 2.0f, { 10, 20 });
	ASSERT_EQ(one.size(), 1u);
	EXPECT_EQ(one[0].id, ref::morton(1, 1, 1));
	EXPECT_EQ(one[0].count, 1u);
	EXPECT_EQ(one[0].occupancy, 30);
	EXPECT_EQ(sized.extract().size(), before.size());
	EXPECT_EQ(sized.voxel_side_length(), 0.5f);
	// the same call twice gives the same answer (no state)
	EXPECT_EQ(sized.voxelize_points(std::vector<Point3f>{ { 3.0f, 3.0f, 3.0f } }, id, 2.0f, { 10, 20 }), one);
}

TEST_F(VulkanTest, OctreeVoxelizeLargeCloud)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Probabilistic);
	std::mt19937_64 rng(77);
	const Matrix4f pose = make_pose(0.1f, 0.2f, 0.3f, 5.0f, 5.0f, 5.0f);
	// 300000 points in about 100000 voxels, and 300000 points that all fall into 8 voxels
	expect_same(tree, random_points(rng, pose, 0.1f, 300000, 60), pose, 0.1f, { 0, 100 }, "many voxels");
	std::vector<Point3f> few;
	std::uniform_int_distribution<uint32_t> d(0, 1);
	for (int i = 0; i < 300000; ++i)
		few.push_back(sensor_point(pose, (5.2 + d(rng)), (5.2 + d(rng)), (5.2 + d(rng))));
	expect_same(tree, few, pose, 1.0f, { 3, -2 }, "few voxels");
}
