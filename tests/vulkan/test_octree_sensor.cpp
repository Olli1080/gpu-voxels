// Slice S6f-4: Octree::insert_point_cloud_with_free_space against the dynamic reference (the inserts the pipeline has to make, in the CUDA order),
// a lookup check of what the pipeline must do to a fresh tree, argument errors without changes to the tree, propagation.
// Slice S6f-3: Octree::pack_free_space is compared with a sparse host reference of the packing (groups found with std::map), alone and behind the rays.
// Slice S6f-2: Octree::cast_free_space_rays is compared with the host translation of the CUDA walk (ProbVoxelMapReference::walk) summed up per cell.
// Slice S6f-1: Octree::voxelize_points, the first step of the sensor pipeline (pose, voxel grid of the pipeline, distinct voxels with the number of points
// and the occupancy of the sensor model). Compared with a plain host reference that works in double precision (std::map of voxel -> count); the test
// points are built from a chosen world position inside a voxel (away from its border) and the inverse pose, so float rounding on the GPU cannot change
// a voxel. Border cases (points exactly on voxel borders, negative zero) use poses with entries 0 / +-1 and side lengths that are powers of two, where
// float arithmetic is exact.
#include "vulkan_fixture.h"

#include <algorithm>
#include <bit>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Octree.h>

#include "../reference/octree_dynamic_reference.h"
#include "../reference/octree_reference.h"
#include "../reference/prob_voxel_map_reference.h"

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
	Expected reference(const std::vector<Point3f>& points, const Matrix4f& pose, float side, const OctreeSensorModel& model, uint32_t limit = 16384)
	{
		std::map<uint64_t, uint32_t> counts;
		Expected e;
		for (const Point3f& p : points)
		{
			double w[3];
			for (int r = 0; r < 3; ++r)
				w[r] = double(pose(r, 0)) * p.x + double(pose(r, 1)) * p.y + double(pose(r, 2)) * p.z + pose(r, 3);
			const double fx = std::floor(w[0] / side), fy = std::floor(w[1] / side), fz = std::floor(w[2] / side);
			if (!(fx >= 0 && fx < limit && fy >= 0 && fy < limit && fz >= 0 && fz < limit))
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

// ---------------------------------------------------------------------------------------------------------------------------------- S6f-2: free-space rays

namespace
{
	using RefWalk = gvl_reference::ProbVoxelMapReference;

	struct RefRays
	{
		uint32_t min[3] = {}, size[3] = {};
		std::vector<uint32_t> counts;
	};

	/// The box of the voxels and the origin, and per cell the number of walks from a voxel to the origin that visit it.
	RefRays reference_rays(const std::vector<OctreeVoxel>& voxels, const OctreeVoxel& origin)
	{
		RefRays r;
		if (voxels.empty())
			return r;
		uint32_t hi[3];
		const uint32_t o[3] = { origin.x, origin.y, origin.z };
		for (int a = 0; a < 3; ++a)
		{
			r.min[a] = hi[a] = o[a];
			for (const OctreeVoxel& v : voxels)
			{
				const uint32_t c = a == 0 ? v.x : a == 1 ? v.y : v.z;
				r.min[a] = std::min(r.min[a], c);
				hi[a] = std::max(hi[a], c);
			}
			r.size[a] = hi[a] - r.min[a] + 1;
		}
		r.counts.assign(size_t(r.size[0]) * r.size[1] * r.size[2], 0);
		for (const OctreeVoxel& v : voxels)
			for (const auto& c : RefWalk::walk({ v.x, v.y, v.z }, { o[0], o[1], o[2] }))
				++r.counts[size_t(c[0] - r.min[0]) + size_t(r.size[0]) * (size_t(c[1] - r.min[1]) + size_t(r.size[1]) * size_t(c[2] - r.min[2]))];
		return r;
	}

	void expect_rays(Octree& tree, const std::vector<OctreeVoxel>& voxels, const OctreeVoxel& origin, const char* what)
	{
		const RefRays e = reference_rays(voxels, origin);
		const OctreeRayMap got = tree.cast_free_space_rays(voxels, origin);
		EXPECT_EQ(got.min.x, e.min[0]) << what;
		EXPECT_EQ(got.min.y, e.min[1]) << what;
		EXPECT_EQ(got.min.z, e.min[2]) << what;
		EXPECT_EQ(got.size.x, e.size[0]) << what;
		EXPECT_EQ(got.size.y, e.size[1]) << what;
		EXPECT_EQ(got.size.z, e.size[2]) << what;
		ASSERT_EQ(got.counts.size(), e.counts.size()) << what;
		for (size_t i = 0; i < e.counts.size(); ++i)
			if (got.counts[i] != e.counts[i])
			{
				ADD_FAILURE() << what << ": cell " << i << " has " << got.counts[i] << " rays, the reference " << e.counts[i];
				break;
			}
	}

	std::vector<OctreeVoxel> random_voxels(std::mt19937_64& rng, size_t count, uint32_t lo, uint32_t extent)
	{
		std::set<std::array<uint32_t, 3>> unique;
		std::uniform_int_distribution<uint32_t> d(lo, lo + extent - 1);
		while (unique.size() < count)
			unique.insert({ d(rng), d(rng), d(rng) });
		std::vector<OctreeVoxel> v;
		for (const auto& c : unique)
			v.push_back({ c[0], c[1], c[2] });
		std::shuffle(v.begin(), v.end(), rng);
		return v;
	}
}

TEST_F(VulkanTest, OctreeRayCastMatchesWalk)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	std::mt19937_64 rng(21);
	for (const uint64_t chunk : { uint64_t(1) << 28, uint64_t(1000), uint64_t(0) })
	{
		tree.test_set_ray_chunk_steps(chunk);
		for (const size_t n : { size_t(1), size_t(2), size_t(10), size_t(300), size_t(3000) })
		{
			for (const uint32_t extent : { 4u, 12u, 60u })
			{
				if (n > size_t(extent) * extent * extent / 2)
					continue;
				const auto voxels = random_voxels(rng, n, 100, extent);
				// the origin inside the box, on its corner, far outside on every side, on the axis of a voxel
				const OctreeVoxel origins[] = { { 100 + extent / 2, 100 + extent / 2, 100 + extent / 2 }, { 100, 100, 100 }, { 100 + extent - 1, 100 + extent - 1, 100 + extent - 1 },
					{ 10, 300, 150 }, { 300, 5, 5 }, { 0, 0, 0 }, { 150, 100, 7 }, { voxels[0].x, voxels[0].y, 3 }, { 100, voxels[0].y, voxels[0].z } };
				for (const OctreeVoxel& o : origins)
					expect_rays(tree, voxels, o, "random rays");
			}
		}
	}
}

TEST_F(VulkanTest, OctreeRayCastKnownRays)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Probabilistic);
	// a ray along an axis visits every cell between the end points, both included
	{
		const auto map = tree.cast_free_space_rays(std::vector<OctreeVoxel>{ { 5, 2, 2 } }, { 0, 2, 2 });
		EXPECT_EQ(map.size.x, 6u);
		EXPECT_EQ(map.size.y, 1u);
		EXPECT_EQ(map.size.z, 1u);
		EXPECT_EQ(map.counts, std::vector<uint32_t>(6, 1));
	}
	// the voxel of the origin alone: one cell, one ray
	{
		const auto map = tree.cast_free_space_rays(std::vector<OctreeVoxel>{ { 7, 8, 9 } }, { 7, 8, 9 });
		EXPECT_EQ(map.counts, std::vector<uint32_t>(1, 1));
		EXPECT_EQ(map.min.x, 7u);
	}
	// a diagonal ray visits 1 + 3 + 3 + 3 cells
	{
		const auto map = tree.cast_free_space_rays(std::vector<OctreeVoxel>{ { 3, 3, 3 } }, { 0, 0, 0 });
		uint32_t sum = 0;
		for (const uint32_t c : map.counts)
			sum += c;
		EXPECT_EQ(sum, 10u);
		EXPECT_EQ(map.counts.front(), 1u); // the origin
		EXPECT_EQ(map.counts.back(), 1u);  // the voxel itself
	}
	// the cells near the origin are visited by every ray: 100 voxels on a line, the origin at one end
	{
		std::vector<OctreeVoxel> line;
		for (uint32_t x = 1; x <= 100; ++x)
			line.push_back({ x, 0, 0 });
		const auto map = tree.cast_free_space_rays(line, { 0, 0, 0 });
		ASSERT_EQ(map.counts.size(), 101u);
		for (uint32_t x = 0; x <= 100; ++x)
			EXPECT_EQ(map.counts[x], 101u - std::max(x, 1u)) << x; // cell x is visited by the rays of the voxels x .. 100, the origin cell by all 100
	}
	// the walk from the voxel to the origin is not the walk from the origin to the voxel: find a pair where the sets differ and check that the GPU takes the first
	{
		auto cells = [](std::array<uint32_t, 3> a, std::array<uint32_t, 3> b)
		{
			std::set<std::array<int64_t, 3>> s;
			for (const auto& c : RefWalk::walk(a, b))
				s.insert(c);
			return s;
		};
		bool found = false;
		for (uint32_t x = 3; x < 12 && !found; ++x)
			for (uint32_t y = 1; y < 12 && !found; ++y)
				for (uint32_t z = 1; z < 12 && !found; ++z)
					if (cells({ x, y, z }, { 0, 0, 0 }) != cells({ 0, 0, 0 }, { x, y, z }))
					{
						const auto map = tree.cast_free_space_rays(std::vector<OctreeVoxel>{ { x, y, z } }, { 0, 0, 0 });
						const auto expected = cells({ x, y, z }, { 0, 0, 0 });
						std::set<std::array<int64_t, 3>> got;
						for (uint32_t k = 0; k < map.counts.size(); ++k)
							if (map.counts[k])
								got.insert({ int64_t(map.min.x + k % map.size.x), int64_t(map.min.y + (k / map.size.x) % map.size.y), int64_t(map.min.z + k / (map.size.x * map.size.y)) });
						EXPECT_EQ(got, expected) << x << " " << y << " " << z;
						found = true;
					}
		EXPECT_TRUE(found) << "no pair with different walks: the check above tests nothing";
	}
}

TEST_F(VulkanTest, OctreeRayCastArgumentsAndLimits)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	// no voxels: nothing is cast, but the origin is checked
	{
		const auto map = tree.cast_free_space_rays({}, { 1, 2, 3 });
		EXPECT_TRUE(map.counts.empty());
		EXPECT_EQ(map.size.x + map.size.y + map.size.z, 0u);
		EXPECT_THROW((void)tree.cast_free_space_rays({}, { 16384, 0, 0 }), VulkanError);
	}
	// coordinates outside the grid
	EXPECT_THROW((void)tree.cast_free_space_rays(std::vector<OctreeVoxel>{ { 1, 1, 1 } }, { 0, 16384, 0 }), VulkanError);
	EXPECT_THROW((void)tree.cast_free_space_rays(std::vector<OctreeVoxel>{ { 1, 1, 16384 } }, { 0, 0, 0 }), VulkanError);
	EXPECT_THROW((void)tree.cast_free_space_rays(std::vector<OctreeVoxel>{ { 1, 1, 1 }, { 0xFFFFFFFFu, 0, 0 } }, { 0, 0, 0 }), VulkanError);
	// the limit of the box: 100 cells pass with a limit of 100, not with 99; the check is before the allocation and the tree stays usable
	const std::vector<OctreeVoxel> row{ { 0, 0, 0 }, { 9, 9, 0 } };
	tree.set_ray_map_limit(100);
	EXPECT_EQ(tree.cast_free_space_rays(row, { 0, 0, 0 }).counts.size(), 100u);
	tree.set_ray_map_limit(99);
	EXPECT_THROW((void)tree.cast_free_space_rays(row, { 0, 0, 0 }), VulkanError);
	tree.set_ray_map_limit(100);
	EXPECT_NO_THROW((void)tree.cast_free_space_rays(row, { 0, 0, 0 }));
	// the whole grid is far above the default limit and is refused without trying to allocate it
	tree.set_ray_map_limit(uint64_t(1) << 28);
	EXPECT_THROW((void)tree.cast_free_space_rays(std::vector<OctreeVoxel>{ { 16383, 16383, 16383 } }, { 0, 0, 0 }), VulkanError);
	// the longest rays that fit into the default limit: along one axis over the whole grid, and a long thin box
	expect_rays(tree, { { 16383, 0, 0 }, { 16383, 3, 2 } }, { 0, 0, 0 }, "grid length");
	expect_rays(tree, { { 0, 16383, 5 }, { 7, 16383, 5 } }, { 3, 0, 0 }, "grid length y");
	// the tree itself does not change
	EXPECT_EQ(tree.extract().size(), 0u);
}

TEST_F(VulkanTest, OctreeRayCastLarge)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	std::mt19937_64 rng(33);
	const auto voxels = random_voxels(rng, 40000, 0, 120);
	expect_rays(tree, voxels, { 60, 60, 60 }, "40000 rays");
	tree.test_set_ray_chunk_steps(100000); // many small submissions give the same map
	expect_rays(tree, voxels, { 2, 100, 7 }, "40000 rays in chunks");
}

TEST_F(VulkanTest, OctreeRayCastChunks)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	std::mt19937_64 rng(44);
	const auto voxels = random_voxels(rng, 100, 50, 12);
	const OctreeVoxel origin{ 40, 60, 70 };
	const RefRays box = reference_rays(voxels, origin);
	const uint64_t longest = uint64_t(box.size[0]) + box.size[1] + box.size[2] - 2; // 1 + dx + dy + dz of the longest ray of the box
	for (const uint64_t chunk : { uint64_t(1) << 28, uint64_t(1000), uint64_t(100), uint64_t(longest), uint64_t(longest - 1), uint64_t(2 * longest), uint64_t(3 * longest),
		uint64_t(3 * longest + longest - 1), uint64_t(20 * longest), uint64_t(0) })
	{
		tree.test_set_ray_chunk_steps(chunk);
		const uint64_t before = tree.test_ray_submissions();
		expect_rays(tree, voxels, origin, "chunks");
		const uint64_t per_chunk = std::max<uint64_t>(1, chunk / longest);
		EXPECT_EQ(tree.test_ray_submissions() - before, (voxels.size() + per_chunk - 1) / per_chunk) << "chunk steps " << chunk;
	}
	// nothing to cast: no submission
	const uint64_t before = tree.test_ray_submissions();
	(void)tree.cast_free_space_rays({}, origin);
	EXPECT_EQ(tree.test_ray_submissions(), before);
}

// ---------------------------------------------------------------------------------------------------------------------------------- S6f-3: packing

namespace
{
	using Cubes = std::vector<std::vector<OctreeFreeCube>>;

	/// The reference of the packing: sparse maps of cell -> value, level by level, groups found with a std::map keyed by the parent cell. The value of a
	/// free cell is max(-10 * count, -127) in a probabilistic tree (from the unclamped count) and 1 in a deterministic one.
	Cubes reference_pack(const OctreeRayMap& map, bool prob)
	{
		Cubes result(Octree::kLevels);
		std::map<std::array<uint32_t, 3>, int> cur;
		for (size_t i = 0; i < map.counts.size(); ++i)
		{
			if (map.counts[i] == 0)
				continue;
			const std::array<uint32_t, 3> c = { map.min.x + uint32_t(i % map.size.x), map.min.y + uint32_t((i / map.size.x) % map.size.y), map.min.z + uint32_t(i / (size_t(map.size.x) * map.size.y)) };
			cur[c] = prob ? int(std::max<int64_t>(-10 * int64_t(map.counts[i]), -127)) : 1;
		}
		for (uint32_t level = 0; level < Octree::kLevels && !cur.empty(); ++level)
		{
			std::map<std::array<uint32_t, 3>, std::vector<std::pair<std::array<uint32_t, 3>, int>>> groups;
			for (const auto& [c, v] : cur)
				groups[{ c[0] >> 1, c[1] >> 1, c[2] >> 1 }].push_back({ c, v });
			std::map<std::array<uint32_t, 3>, int> next;
			for (const auto& [parent, children] : groups)
			{
				bool equal = true;
				for (const auto& child : children)
					equal = equal && child.second == children[0].second;
				if (children.size() == 8 && equal)
					next[parent] = children[0].second;
				else
					for (const auto& [c, v] : children)
						result[level].push_back({ ref::morton(c[0], c[1], c[2]), static_cast<int8_t>(prob ? v : 0) });
			}
			std::sort(result[level].begin(), result[level].end(), [](const OctreeFreeCube& a, const OctreeFreeCube& b) { return a.id < b.id; });
			cur = std::move(next);
		}
		return result;
	}

	void expect_cubes(Octree& tree, const OctreeRayMap& map, bool prob, const char* what)
	{
		const Cubes e = reference_pack(map, prob);
		const Cubes got = tree.pack_free_space(map);
		ASSERT_EQ(got.size(), e.size()) << what;
		for (size_t l = 0; l < e.size(); ++l)
		{
			EXPECT_EQ(got[l].size(), e[l].size()) << what << ", level " << l;
			for (size_t i = 0; i < std::min(got[l].size(), e[l].size()); ++i)
				if (!(got[l][i] == e[l][i]))
				{
					ADD_FAILURE() << what << ", level " << l << ", cube " << i << ": (id " << got[l][i].id << ", value " << int(got[l][i].occupancy) << ") but the reference has (id " << e[l][i].id << ", value "
						<< int(e[l][i].occupancy) << ")";
					break;
				}
		}
	}

	OctreeRayMap make_map(OctreeVoxel min, OctreeVoxel size, std::vector<uint32_t> counts)
	{
		OctreeRayMap m;
		m.min = min;
		m.size = size;
		m.counts = std::move(counts);
		return m;
	}

	/// Random counts: a cell is free with probability `density`, its count is 1 .. max_count.
	OctreeRayMap random_map(std::mt19937_64& rng, OctreeVoxel min, OctreeVoxel size, double density, uint32_t max_count)
	{
		std::bernoulli_distribution free(density);
		std::uniform_int_distribution<uint32_t> count(1, max_count);
		std::vector<uint32_t> counts(size_t(size.x) * size.y * size.z, 0);
		for (auto& c : counts)
			if (free(rng))
				c = count(rng);
		return make_map(min, size, std::move(counts));
	}
}

TEST_F(VulkanTest, OctreePackFreeSpaceMatchesReference)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(55);
	for (const OctreeKind kind : kKinds)
	{
		Octree tree(ctx(), kind);
		const bool prob = kind == OctreeKind::Probabilistic;
		// boxes of every alignment, the density from sparse to full, counts from all equal to widely different (and above the clamp of 13)
		for (const OctreeVoxel min : { OctreeVoxel{ 0, 0, 0 }, OctreeVoxel{ 1, 0, 3 }, OctreeVoxel{ 3, 5, 7 }, OctreeVoxel{ 6, 6, 6 }, OctreeVoxel{ 13, 22, 9 }, OctreeVoxel{ 16321, 16300, 16370 } })
			for (const OctreeVoxel size : { OctreeVoxel{ 1, 1, 1 }, OctreeVoxel{ 2, 2, 2 }, OctreeVoxel{ 5, 3, 4 }, OctreeVoxel{ 8, 8, 8 }, OctreeVoxel{ 9, 17, 12 }, OctreeVoxel{ 30, 30, 30 }, OctreeVoxel{ 64, 3, 40 } })
				for (const double density : { 0.05, 0.5, 0.9, 0.99, 1.0 })
					for (const uint32_t max_count : { 1u, 2u, 14u, 40u })
					{
						if (min.x + size.x > 16384 || min.y + size.y > 16384 || min.z + size.z > 16384)
							continue;
						expect_cubes(tree, random_map(rng, min, size, density, max_count), prob, "random map");
					}
	}
}

TEST_F(VulkanTest, OctreePackFreeSpaceKnownCases)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree det(ctx(), OctreeKind::Deterministic), prob(ctx(), OctreeKind::Probabilistic);
	// one full 8x8x8 box on an aligned corner: a single cube of level 3, for both kinds, with the value -10 in the probabilistic tree
	{
		const auto map = make_map({ 8, 16, 24 }, { 8, 8, 8 }, std::vector<uint32_t>(512, 1));
		for (Octree* tree : { &det, &prob })
		{
			const auto cubes = tree->pack_free_space(map);
			ASSERT_EQ(cubes.size(), size_t(Octree::kLevels));
			for (uint32_t l = 0; l < Octree::kLevels; ++l)
				if (l != 3)
					EXPECT_TRUE(cubes[l].empty()) << l;
			ASSERT_EQ(cubes[3].size(), 1u);
			EXPECT_EQ(cubes[3][0].id, ref::morton(1, 2, 3));
			EXPECT_EQ(cubes[3][0].occupancy, tree == &prob ? -10 : 0);
		}
	}
	// the same box shifted by one cell: no aligned 8-cube exists any more, the packing finds the aligned parts
	{
		const auto cubes = det.pack_free_space(make_map({ 9, 16, 24 }, { 8, 8, 8 }, std::vector<uint32_t>(512, 1)));
		uint64_t voxels = 0;
		for (uint32_t l = 0; l < Octree::kLevels; ++l)
			voxels += uint64_t(cubes[l].size()) << (3 * l);
		EXPECT_EQ(voxels, 512u); // the cubes cover exactly the free cells
		EXPECT_TRUE(cubes[3].empty());
	}
	// 2 x 2 x 2: the deterministic tree packs, the probabilistic one only if the counts are equal (below and above the clamp)
	{
		std::vector<uint32_t> counts(8, 1);
		counts[7] = 2;
		EXPECT_EQ(det.pack_free_space(make_map({ 4, 4, 4 }, { 2, 2, 2 }, counts))[1].size(), 1u);
		const auto split = prob.pack_free_space(make_map({ 4, 4, 4 }, { 2, 2, 2 }, counts));
		EXPECT_TRUE(split[1].empty());
		EXPECT_EQ(split[0].size(), 8u);
		EXPECT_EQ(split[0][7].occupancy, -20);
		counts.assign(8, 13);
		counts[3] = 400; // 13 rays and 400 rays both give -127
		const auto clamped = prob.pack_free_space(make_map({ 4, 4, 4 }, { 2, 2, 2 }, counts));
		ASSERT_EQ(clamped[1].size(), 1u);
		EXPECT_EQ(clamped[1][0].occupancy, -127);
		EXPECT_TRUE(clamped[0].empty());
	}
	// one hole in a 4x4x4 block: the other seven 2x2x2 groups pack, the group with the hole stays as 7 cells
	{
		std::vector<uint32_t> counts(64, 1);
		counts[0] = 0;
		const auto cubes = det.pack_free_space(make_map({ 0, 0, 0 }, { 4, 4, 4 }, counts));
		EXPECT_EQ(cubes[0].size(), 7u);
		EXPECT_EQ(cubes[1].size(), 7u);
		EXPECT_TRUE(cubes[2].empty());
	}
	// packing over many levels: a full 64 cube on the top corner of the grid is one cube of level 6
	{
		const auto cubes = det.pack_free_space(make_map({ 16320, 16320, 16320 }, { 64, 64, 64 }, std::vector<uint32_t>(64 * 64 * 64, 3)));
		for (uint32_t l = 0; l < Octree::kLevels; ++l)
			EXPECT_EQ(cubes[l].size(), l == 6 ? 1u : 0u) << l;
		EXPECT_EQ(cubes[6][0].id, ref::morton(255, 255, 255));
	}
	// no free cell at all, and the empty map
	EXPECT_EQ(det.pack_free_space(make_map({ 5, 5, 5 }, { 3, 3, 3 }, std::vector<uint32_t>(27, 0))), Cubes(Octree::kLevels));
	EXPECT_EQ(det.pack_free_space(OctreeRayMap{}), Cubes(Octree::kLevels));
}

TEST_F(VulkanTest, OctreePackFreeSpaceBigBox)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Probabilistic);
	// 128 cells per side all free with 5 rays each: one cube of level 7 (the packing runs over 7 levels, 2M cells at the first)
	const auto cubes = tree.pack_free_space(make_map({ 256, 128, 384 }, { 128, 128, 128 }, std::vector<uint32_t>(128 * 128 * 128, 5)));
	for (uint32_t l = 0; l < Octree::kLevels; ++l)
		EXPECT_EQ(cubes[l].size(), l == 7 ? 1u : 0u) << l;
	EXPECT_EQ(cubes[7][0].id, ref::morton(2, 1, 3));
	EXPECT_EQ(cubes[7][0].occupancy, -50);
}

TEST_F(VulkanTest, OctreePackFreeSpaceArguments)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	// the counts must fit the box, the box must lie in the grid and have cells
	EXPECT_THROW((void)tree.pack_free_space(make_map({ 0, 0, 0 }, { 2, 2, 2 }, std::vector<uint32_t>(7, 1))), VulkanError);
	EXPECT_THROW((void)tree.pack_free_space(make_map({ 0, 0, 0 }, { 2, 2, 2 }, std::vector<uint32_t>(9, 1))), VulkanError);
	EXPECT_THROW((void)tree.pack_free_space(make_map({ 16380, 0, 0 }, { 8, 1, 1 }, std::vector<uint32_t>(8, 1))), VulkanError);
	EXPECT_NO_THROW((void)tree.pack_free_space(make_map({ 16380, 0, 0 }, { 4, 1, 1 }, std::vector<uint32_t>(4, 1))));
	EXPECT_THROW((void)tree.pack_free_space(make_map({ 16384, 0, 0 }, { 1, 1, 1 }, std::vector<uint32_t>(1, 1))), VulkanError);
	EXPECT_THROW((void)tree.pack_free_space(make_map({ 20000, 0, 0 }, { 1, 1, 1 }, std::vector<uint32_t>(1, 1))), VulkanError);
	EXPECT_THROW((void)tree.pack_free_space(make_map({ 0, 0, 0xFFFFFFF0u }, { 1, 1, 1 }, std::vector<uint32_t>(1, 1))), VulkanError);
	EXPECT_THROW((void)tree.pack_free_space(make_map({ 0, 0, 0 }, { 0, 2, 2 }, {})), VulkanError);
	// the limit counts the cells of the box extended to even coordinates: one cell at an odd position takes 2 x 2 x 2
	const auto one = make_map({ 3, 3, 3 }, { 1, 1, 1 }, std::vector<uint32_t>(1, 1));
	tree.set_ray_map_limit(7);
	EXPECT_THROW((void)tree.pack_free_space(one), VulkanError);
	tree.set_ray_map_limit(8);
	EXPECT_NO_THROW((void)tree.pack_free_space(one));
	// nothing changes after a refusal
	EXPECT_EQ(tree.pack_free_space(one)[0].size(), 1u);
}

TEST_F(VulkanTest, OctreeRaysAndPacking)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(66);
	for (const OctreeKind kind : kKinds)
	{
		Octree tree(ctx(), kind);
		for (const size_t n : { size_t(5), size_t(200), size_t(3000) })
			for (const uint32_t extent : { 10u, 24u, 40u })
			{
				if (n > size_t(extent) * extent * extent / 4)
					continue;
				const auto voxels = random_voxels(rng, n, 200, extent);
				for (const OctreeVoxel& o : { OctreeVoxel{ 200 + extent / 2, 200 + extent / 2, 200 + extent / 2 }, OctreeVoxel{ 190, 215, 230 }, OctreeVoxel{ 199, 199, 199 } })
				{
					const RefRays rays = reference_rays(voxels, o);
					OctreeRayMap map;
					map.min = { rays.min[0], rays.min[1], rays.min[2] };
					map.size = { rays.size[0], rays.size[1], rays.size[2] };
					map.counts = rays.counts;
					expect_cubes(tree, map, kind == OctreeKind::Probabilistic, "rays then packing");
					// the same through the GPU rays
					const OctreeRayMap gpu = tree.cast_free_space_rays(voxels, o);
					EXPECT_EQ(tree.pack_free_space(gpu), tree.pack_free_space(map));
				}
			}
	}
}

// ---------------------------------------------------------------------------------------------------------------------------------- S6f-4: the whole pipeline

namespace
{
	using Dyn = ref::DynamicOctreeReference;
	ref::Kind to_ref_kind(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? ref::Kind::Probabilistic : ref::Kind::Deterministic; }

	std::string difference(const std::vector<OctreeNode>& gpu, const std::vector<ref::Node>& expected)
	{
		if (gpu.size() != expected.size())
			return "node counts differ: GPU " + std::to_string(gpu.size()) + ", reference " + std::to_string(expected.size());
		for (size_t i = 0; i < gpu.size(); ++i)
		{
			const auto& a = gpu[i];
			const auto& b = expected[i];
			if (a.level != b.level || a.id != b.id || a.status != b.status || a.occupancy != b.occupancy || a.has_children != b.has_children)
				return "node " + std::to_string(i) + ": GPU (level " + std::to_string(a.level) + ", id " + std::to_string(a.id) + ", status " + std::to_string(a.status) + ", occupancy " +
					std::to_string(a.occupancy) + (a.has_children ? ", children" : "") + ") reference (level " + std::to_string(b.level) + ", id " + std::to_string(b.id) + ", status " +
					std::to_string(b.status) + ", occupancy " + std::to_string(b.occupancy) + (b.has_children ? ", children" : "") + ")";
		}
		return {};
	}

	void expect_tree(const Octree& tree, const Dyn& model, const std::string& what)
	{
		EXPECT_EQ(difference(tree.dump(), model.dump()), "") << what;
		EXPECT_NO_THROW(tree.check_tree()) << what;
		EXPECT_EQ(tree.pending_update_count(), 0u) << what << ": NEEDS_UPDATE left behind";
		const auto stats = tree.memory_usage();
		for (uint32_t level = 0; level < 14; ++level)
			EXPECT_EQ(stats.garbage_blocks[level], model.garbage_blocks(level)) << what << ", garbage blocks of level " << level;
	}

	OctreeVoxel voxel_of_id(uint64_t id)
	{
		OctreeVoxel v;
		for (uint32_t b = 0; b < 14; ++b)
		{
			v.x |= uint32_t((id >> (3 * b)) & 1) << b;
			v.y |= uint32_t((id >> (3 * b + 1)) & 1) << b;
			v.z |= uint32_t((id >> (3 * b + 2)) & 1) << b;
		}
		return v;
	}

	/// A sensor with a cloud around it: the world positions lie on a spherical shell, each inside a voxel of the TREE grid and away from its border (which
	/// keeps it away from the borders of every coarser grid too), so float and double arithmetic give the same voxels.
	struct Scene
	{
		std::vector<Point3f> points;
		Matrix4f pose;
	};

	/// `skewed`: the sensor has different coordinates on the axes (otherwise the same on all three, for scenes near the far end of the grid).
	Scene make_scene(std::mt19937_64& rng, float side, double center_voxel, double radius_lo, double radius_hi, size_t n, bool skewed = true)
	{
		Scene s;
		std::uniform_real_distribution<double> angle(-3.0, 3.0), frac(0.25, 0.75), radius(radius_lo, radius_hi), unit(-1.0, 1.0);
		const double c = (std::floor(center_voxel) + 0.5) * side; // the sensor in the middle of a voxel
		s.pose = make_pose(float(angle(rng)), float(angle(rng)), float(angle(rng)), float(c), float(skewed ? c * 1.1 : c), float(skewed ? c * 0.9 : c));
		for (size_t i = 0; i < n; ++i)
		{
			double d[3] = { unit(rng), unit(rng), unit(rng) };
			const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
			if (len < 1e-3)
				continue;
			const double r = radius(rng);
			double w[3];
			const double origin[3] = { double(s.pose(0, 3)), double(s.pose(1, 3)), double(s.pose(2, 3)) };
			for (int a = 0; a < 3; ++a)
				w[a] = (std::floor((origin[a] + r * d[a] / len) / side) + frac(rng)) * side;
			s.points.push_back(sensor_point(s.pose, w[0], w[1], w[2]));
			if (i % 4 == 3)
				s.points.push_back(s.points[rng() % s.points.size()]); // repeated points: counts above 1
		}
		return s;
	}

	/// The sensor pipeline as the reference sees it (the pieces were checked one by one above): voxelize both grids, rays, packing, the inserts of the
	/// dynamic reference in the CUDA order, propagate.
	struct SensorExpectation
	{
		Expected objects, free_space;
		OctreeVoxel origin;
		std::vector<OctreeVoxel> free_voxels;
		RefRays rays;
		Cubes cubes;
	};

	SensorExpectation expect_sensor(const Scene& s, float side, const OctreeSensorOptions& o, bool prob)
	{
		SensorExpectation e;
		const uint32_t fs = uint32_t(std::countr_zero(o.free_space_scale)), os = uint32_t(std::countr_zero(o.object_scale));
		e.objects = reference(s.points, s.pose, side * float(o.object_scale), o.object_model, 16384u >> os);
		e.free_space = reference(s.points, s.pose, side * float(o.free_space_scale), {}, 16384u >> fs);
		uint32_t* axis[3] = { &e.origin.x, &e.origin.y, &e.origin.z };
		for (int a = 0; a < 3; ++a)
			*axis[a] = uint32_t(std::floor(double(s.pose(a, 3)) / (double(side) * o.free_space_scale)));
		for (const auto& run : e.free_space.runs)
			e.free_voxels.push_back(voxel_of_id(run.id));
		OctreeRayMap map;
		if (!e.free_voxels.empty())
		{
			e.rays = reference_rays(e.free_voxels, e.origin);
			map.min = { e.rays.min[0], e.rays.min[1], e.rays.min[2] };
			map.size = { e.rays.size[0], e.rays.size[1], e.rays.size[2] };
			map.counts = e.rays.counts;
		}
		e.cubes = reference_pack(map, prob);
		return e;
	}

	void apply_to_reference(Dyn& model, const SensorExpectation& e, const OctreeSensorOptions& o, bool prob)
	{
		const uint32_t fs = uint32_t(std::countr_zero(o.free_space_scale)), os = uint32_t(std::countr_zero(o.object_scale));
		for (uint32_t l = 0; l < e.cubes.size(); ++l)
			for (const OctreeFreeCube& cube : e.cubes[l])
				model.insert(cube.id << (3 * (l + fs)), Dyn::Datum::Free, l + fs, Dyn::Mode::Soft, prob ? int(cube.occupancy) : Dyn::kDatumValue);
		for (const OctreeVoxelRun& run : e.objects.runs)
			model.insert(run.id << (3 * os), Dyn::Datum::Occupied, os, Dyn::Mode::Hard, prob ? int(run.occupancy) : Dyn::kDatumValue);
		model.propagate();
	}
}

TEST_F(VulkanTest, OctreeSensorInsertMatchesReference)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(88);
	const float side = 0.05f;
	const OctreeSensorModel models[] = { { 0, 100 }, { 0, 50 }, { 20, 30 }, { 0, -5 } };
	for (const OctreeKind kind : kKinds)
	{
		const bool prob = kind == OctreeKind::Probabilistic;
		for (const auto [free_scale, object_scale] : { std::pair{ 1u, 1u }, std::pair{ 2u, 1u }, std::pair{ 4u, 2u }, std::pair{ 1u, 2u }, std::pair{ 8u, 4u }, std::pair{ 2u, 2u } })
		{
			Octree tree(ctx(), kind, side);
			Dyn model(to_ref_kind(kind));
			OctreeSensorOptions options;
			options.free_space_scale = free_scale;
			options.object_scale = object_scale;
			options.object_model = models[rng() % std::size(models)];
			// three clouds from different poses: the second and third insert into an existing tree (existing leaves and nodes, soft inserts into PART nodes)
			for (const size_t n : { size_t(300), size_t(1500), size_t(600) })
			{
				const Scene scene = make_scene(rng, side, 60.0 + double(rng() % 40), 0.6, 2.0, n);
				const SensorExpectation e = expect_sensor(scene, side, options, prob);
				const OctreeSensorResult r = tree.insert_point_cloud_with_free_space(scene.points, scene.pose, options);
				apply_to_reference(model, e, options, prob);
				const std::string what = std::string(prob ? "probabilistic" : "deterministic") + " scales " + std::to_string(free_scale) + "/" + std::to_string(object_scale) + ", " + std::to_string(n) + " points";
				expect_tree(tree, model, what);
				EXPECT_EQ(r.object_voxels, e.objects.runs.size()) << what;
				EXPECT_EQ(r.free_voxels, e.free_space.runs.size()) << what;
				EXPECT_EQ(r.outside_object, e.objects.outside) << what;
				EXPECT_EQ(r.outside_free, e.free_space.outside) << what;
				uint64_t cubes = 0;
				for (const auto& level : e.cubes)
					cubes += level.size();
				EXPECT_EQ(r.free_cubes, cubes) << what;
				EXPECT_GT(r.free_cubes, 0u) << what << ": the scene has free space";
			}
			// far from the origin of the grid: ids with bits above 2^32, a sensor at coordinates up to 15000 voxels
			for (const double center : { 1200.0, 9000.0, 15000.0 })
			{
				const Scene scene = make_scene(rng, side, center + double(rng() % 200), 0.6, 2.0, 400, false);
				const SensorExpectation e = expect_sensor(scene, side, options, prob);
				tree.insert_point_cloud_with_free_space(scene.points, scene.pose, options);
				apply_to_reference(model, e, options, prob);
				expect_tree(tree, model, std::string(prob ? "probabilistic" : "deterministic") + " scales " + std::to_string(free_scale) + "/" + std::to_string(object_scale) + ", far scene at " + std::to_string(center));
			}
		}
	}
}

TEST_F(VulkanTest, OctreeSensorInsertMeaning)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	// What a point cloud with free space has to do to a fresh tree (scales 1 / 1), checked with lookups and without the reference of the packing: the cells
	// the rays visit are free (probabilistic: the value of their rays), an object voxel is occupied (probabilistic: the value of its free cell plus the
	// sensor model, because the object is inserted into the leaf the free space made), everything else is unknown.
	std::mt19937_64 rng(99);
	const float side = 0.05f;
	for (const OctreeKind kind : kKinds)
	{
		const bool prob = kind == OctreeKind::Probabilistic;
		Octree tree(ctx(), kind, side);
		OctreeSensorOptions options;
		options.object_model = { 0, 100 };
		const Scene scene = make_scene(rng, side, 70.0, 0.5, 1.5, 400);
		const SensorExpectation e = expect_sensor(scene, side, options, prob);
		tree.insert_point_cloud_with_free_space(scene.points, scene.pose, options);
		std::map<uint64_t, uint32_t> object_count;
		std::map<uint64_t, int> object_occupancy;
		for (const auto& run : e.objects.runs)
		{
			object_count[run.id] = run.count;
			object_occupancy[run.id] = run.occupancy;
		}
		std::vector<uint64_t> ids;
		std::vector<std::pair<int, int>> expected; // free ray count, object occupancy (-1000 = none)
		for (size_t i = 0; i < e.rays.counts.size(); ++i)
		{
			const OctreeVoxel v{ e.rays.min[0] + uint32_t(i % e.rays.size[0]), e.rays.min[1] + uint32_t((i / e.rays.size[0]) % e.rays.size[1]), e.rays.min[2] + uint32_t(i / (size_t(e.rays.size[0]) * e.rays.size[1])) };
			const uint64_t id = ref::morton(v.x, v.y, v.z);
			if (e.rays.counts[i] == 0 && !object_count.count(id))
				continue;
			ids.push_back(id);
			expected.push_back({ int(e.rays.counts[i]), object_count.count(id) ? object_occupancy[id] : -1000 });
		}
		for (const auto& run : e.objects.runs) // object voxels outside the ray box
			if (std::find(ids.begin(), ids.end(), run.id) == ids.end())
			{
				ids.push_back(run.id);
				expected.push_back({ 0, run.occupancy });
			}
		const auto found = tree.find(ids);
		ASSERT_EQ(found.size(), ids.size());
		size_t occupied = 0, freed = 0;
		for (size_t i = 0; i < ids.size(); ++i)
		{
			const auto [rays, object] = expected[i];
			if (prob)
			{
				const int free_value = rays > 0 ? std::max(-10 * rays, -127) : 0;
				int want;
				if (object != -1000)
					want = rays > 0 ? std::clamp(free_value + object, -127, 127) : object;
				else
					want = free_value;
				EXPECT_EQ(int(found[i].occupancy), want) << "voxel " << ids[i] << ", rays " << rays << ", object " << object;
			}
			else
			{
				const uint8_t want = object != -1000 ? octree_format::kOccupied : octree_format::kFree;
				EXPECT_EQ(found[i].status & octree_format::kOccupancyMask, want) << "voxel " << ids[i] << ", rays " << rays << ", object " << object;
			}
			object != -1000 ? ++occupied : ++freed;
		}
		EXPECT_GT(occupied, 10u);
		EXPECT_GT(freed, 100u);
		// voxels no ray reached and no point is in stay unknown
		const std::vector<uint64_t> far_away = { ref::morton(1, 1, 1), ref::morton(2000, 2000, 2000), ref::morton(16383, 16383, 16383) };
		for (const auto& f : tree.find(far_away))
			EXPECT_EQ(f.status & octree_format::kOccupancyMask, octree_format::kUnknown);
	}
}

TEST_F(VulkanTest, OctreeSensorInsertArguments)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(111);
	const float side = 0.05f;
	for (const OctreeKind kind : kKinds)
	{
		const bool prob = kind == OctreeKind::Probabilistic;
		const Scene scene = make_scene(rng, side, 60.0, 0.6, 1.5, 300);
		OctreeSensorOptions options;
		options.object_model = { 0, 100 };
		// no voxel side length: the message says so (the origin check would refuse the pose with another message), also without points
		{
			Octree tree(ctx(), kind);
			for (const std::vector<Point3f>& points : { scene.points, std::vector<Point3f>{} })
			{
				try
				{
					(void)tree.insert_point_cloud_with_free_space(points, scene.pose, options);
					ADD_FAILURE() << "no exception";
				}
				catch (const VulkanError& e)
				{
					EXPECT_NE(std::string(e.what()).find("side length"), std::string::npos) << e.what();
				}
			}
		}
		Octree tree(ctx(), kind, side);
		const auto empty_dump = tree.dump();
		// scales that are not powers of two, zero, above 2^13: refused, nothing inserted
		for (const uint32_t bad : { 0u, 3u, 6u, 16384u, 1u << 14, 1u << 20 })
		{
			OctreeSensorOptions o = options;
			o.free_space_scale = bad;
			EXPECT_THROW((void)tree.insert_point_cloud_with_free_space(scene.points, scene.pose, o), VulkanError) << bad;
			o = options;
			o.object_scale = bad;
			EXPECT_THROW((void)tree.insert_point_cloud_with_free_space(scene.points, scene.pose, o), VulkanError) << bad;
		}
		// the sensor outside the grid of the free space (negative, too far; the limit depends on the scale), nothing inserted
		for (const auto [x, scale] : { std::pair{ -0.5f, 1u }, std::pair{ -1e-3f, 1u }, std::pair{ 16384.0f * side, 1u }, std::pair{ 16384.0f * side - 0.01f, 2u + 0u }, std::pair{ 8192.0f * 2 * side, 2u } })
		{
			OctreeSensorOptions o = options;
			o.free_space_scale = scale;
			Matrix4f pose = scene.pose;
			pose(0, 3) = x;
			const bool valid = x >= 0 && std::floor(double(x) / (double(side) * scale)) < double(16384 / scale);
			if (valid)
				continue; // the one valid position in the list is not an error
			EXPECT_THROW((void)tree.insert_point_cloud_with_free_space(scene.points, pose, o), VulkanError) << x << " scale " << scale;
		}
		EXPECT_TRUE(difference(tree.dump(), ref::DynamicOctreeReference(to_ref_kind(kind)).dump()).empty());
		EXPECT_EQ(tree.dump().size(), empty_dump.size());
		// a ray box above the limit is detected before the tree changes
		tree.set_ray_map_limit(100);
		EXPECT_THROW((void)tree.insert_point_cloud_with_free_space(scene.points, scene.pose, options), VulkanError);
		EXPECT_EQ(tree.dump().size(), empty_dump.size());
		EXPECT_EQ(tree.pending_update_count(), 0u);
		tree.set_ray_map_limit(uint64_t(1) << 28);
		// no points: nothing happens, also with propagate
		{
			const OctreeSensorResult r = tree.insert_point_cloud_with_free_space({}, scene.pose, options);
			EXPECT_EQ(r.object_voxels + r.free_voxels + r.outside_object + r.outside_free, 0u);
			EXPECT_EQ(r.free_cubes, 0u);
			EXPECT_EQ(tree.dump().size(), empty_dump.size());
		}
		// the checks of the arguments do not depend on the points: a sensor outside the grid and a scale that is not allowed are refused for an empty cloud too
		{
			Matrix4f negative = scene.pose;
			negative(0, 3) = -1.0f;
			EXPECT_THROW((void)tree.insert_point_cloud_with_free_space({}, negative, options), VulkanError);
			for (const uint32_t bad : { 0u, 3u, 1u << 14 })
			{
				OctreeSensorOptions o = options;
				o.object_scale = bad;
				EXPECT_THROW((void)tree.insert_point_cloud_with_free_space({}, scene.pose, o), VulkanError) << bad;
				o = options;
				o.free_space_scale = bad;
				EXPECT_THROW((void)tree.insert_point_cloud_with_free_space({}, scene.pose, o), VulkanError) << bad;
			}
		}
		// points outside the grid: counted, not inserted. Every grid covers the extent of the tree (a coarse cell is a block of tree voxels), so the last voxel of the
		// tree grid is inside the coarser free-space grid too and both counts agree
		{
			const float nan = std::numeric_limits<float>::quiet_NaN();
			std::vector<Point3f> points = scene.points;
			const Matrix4f id = Matrix4f::identity();
			(void)id;
			// the points in world coordinates through the inverse pose
			points.push_back(sensor_point(scene.pose, -1.0, 2.0, 2.0));
			points.push_back(sensor_point(scene.pose, 3.0, -1.0, 2.0));
			points.push_back({ nan, 1.0f, 1.0f });
			points.push_back(sensor_point(scene.pose, 16383.5 * side, 3.0, 3.0)); // the last voxel of the tree grid in x
			points.push_back(sensor_point(scene.pose, 16384.5 * side, 3.0, 3.0)); // one voxel behind it: outside the coarse grid too, whose limit is 8192 cells
			OctreeSensorOptions o = options;
			o.free_space_scale = 2;
			const SensorExpectation e = expect_sensor(Scene{ points, scene.pose }, side, o, prob);
			const OctreeSensorResult r = tree.insert_point_cloud_with_free_space(points, scene.pose, o);
			EXPECT_EQ(r.outside_object, e.objects.outside);
			EXPECT_EQ(r.outside_free, e.free_space.outside);
			EXPECT_EQ(r.outside_object, 4u);
			EXPECT_EQ(r.outside_free, 4u);
			EXPECT_NO_THROW(tree.check_tree());
		}
		// the same with a coarse grid of the objects: one voxel behind the edge is outside its limit of 8192 cells and must not be inserted
		{
			std::vector<Point3f> points = scene.points;
			points.push_back(sensor_point(scene.pose, 16384.5 * side, 3.0, 3.0));
			points.push_back(sensor_point(scene.pose, 16383.5 * side, 3.0, 3.0)); // the last cell of the coarse grid
			OctreeSensorOptions o = options;
			o.object_scale = 2;
			const SensorExpectation e = expect_sensor(Scene{ points, scene.pose }, side, o, prob);
			const OctreeSensorResult r = tree.insert_point_cloud_with_free_space(points, scene.pose, o);
			EXPECT_EQ(r.outside_object, e.objects.outside);
			EXPECT_EQ(r.outside_object, 1u);
			EXPECT_EQ(r.outside_free, 1u);
			EXPECT_NO_THROW(tree.check_tree());
		}
	}
}

TEST_F(VulkanTest, OctreeSensorInsertPropagation)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	std::mt19937_64 rng(123);
	const float side = 0.05f;
	for (const OctreeKind kind : kKinds)
	{
		const Scene a = make_scene(rng, side, 65.0, 0.6, 1.8, 800), b = make_scene(rng, side, 70.0, 0.6, 1.8, 800);
		OctreeSensorOptions options;
		options.free_space_scale = 2;
		options.object_model = { 0, 100 };
		Octree now(ctx(), kind, side), later(ctx(), kind, side);
		now.insert_point_cloud_with_free_space(a.points, a.pose, options);
		now.insert_point_cloud_with_free_space(b.points, b.pose, options);
		// the same inserts with the propagation done separately, once after each call: the same tree
		later.insert_point_cloud_with_free_space(a.points, a.pose, options, false);
		EXPECT_GT(later.pending_update_count(), 0u);
		later.propagate();
		later.insert_point_cloud_with_free_space(b.points, b.pose, options, false);
		later.propagate();
		EXPECT_TRUE(now.dump() == later.dump());
		EXPECT_EQ(later.pending_update_count(), 0u);
		EXPECT_NO_THROW(later.check_tree());
	}
}

TEST_F(VulkanTest, OctreeSensorInsertFreeSpaceOverDetail)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	// Free space of a coarse grid over a tree that already has detail: the soft insert of a cube on a node with PART marks UPDATE_SUBTREE, so the propagate
	// has to reach many more nodes than there are inserted ids and must not take the bounded path (its lists are sized from the number of inserted ids).
	std::mt19937_64 rng(131);
	const float side = 0.05f;
	for (const OctreeKind kind : kKinds)
	{
		Octree tree(ctx(), kind, side);
		Dyn model(to_ref_kind(kind));
		const bool prob = kind == OctreeKind::Probabilistic;
		OctreeSensorOptions fine;
		fine.object_model = { 0, 100 };
		const Scene detail = make_scene(rng, side, 80.0, 0.6, 1.4, 3000); // a dense shell: detail on every level
		apply_to_reference(model, expect_sensor(detail, side, fine, prob), fine, prob);
		tree.insert_point_cloud_with_free_space(detail.points, detail.pose, fine);
		expect_tree(tree, model, "detail");
		OctreeSensorOptions coarse = fine;
		coarse.free_space_scale = 8; // the free cells are nodes of level 3
		// the same sensor position (the translation depends on the centre only), points farther out: the rays cross the detailed shell
		const Scene outer = make_scene(rng, side, 80.0, 1.6, 2.0, 1500);
		const SensorExpectation e = expect_sensor(outer, side, coarse, prob);
		const OctreeSensorResult r = tree.insert_point_cloud_with_free_space(outer.points, outer.pose, coarse);
		apply_to_reference(model, e, coarse, prob);
		expect_tree(tree, model, "free space over detail");
		EXPECT_GT(r.free_cubes, 50u);
	}
}
