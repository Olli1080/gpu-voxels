// Slice S7-5: headless versions of examples of packages/examples as smoke tests. Each test repeats the GpuVoxels call sequence of one example
// without the viewer and the endless loop (a few frames instead), and checks the results against numbers that do not come from the call under
// test: hand-calculated counts, the voxel formula in double precision, or host models built from the downloaded states of the maps and lists.
// Differences to the programs are noted at each test. Not translated (and why): see Agent/knowledge_base/architecture/vulkan_gpu_voxels_examples.md.
#include "vulkan_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>

#include "../reference/prob_voxel_map_reference.h"
#include "list_collision_model.h"

using namespace gpu_voxels::vulkan;
using namespace gvl_test::collision_model;
using gvl_test::VulkanTest;

namespace
{
	const std::filesystem::path kModels = GVL_TEST_MODEL_DIR;

	/// The voxel range of one box axis as CUDA createBoxOfPoints computes it, in float arithmetic: floor(max(min, 0) / side) .. ceil(max / side) inclusive.
	struct AxisRange
	{
		float first, last;
	};

	AxisRange axis_range(float lo, float hi, float side) { return { std::floor((lo >= 0.0f ? lo : 0.0f) / side), std::ceil(hi / side) }; }

	/// Number of voxels of insert_box_into_map (the box of the example lies inside the map).
	uint64_t box_voxels(Point3f min, Point3f max, float side)
	{
		const auto axis = [side](float lo, float hi) { const AxisRange r = axis_range(lo, hi, side); return static_cast<uint64_t>(r.last - r.first + 1.0f); };
		return axis(min.x, max.x) * axis(min.y, max.y) * axis(min.z, max.z);
	}

	/// Voxels that both boxes cover (same voxel ranges as box_voxels).
	uint64_t box_overlap(Point3f min_a, Point3f max_a, Point3f min_b, Point3f max_b, float side)
	{
		const auto axis = [side](float lo_a, float hi_a, float lo_b, float hi_b)
		{
			const AxisRange a = axis_range(lo_a, hi_a, side), b = axis_range(lo_b, hi_b, side);
			const float lo = std::max(a.first, b.first), hi = std::min(a.last, b.last);
			return hi >= lo ? static_cast<uint64_t>(hi - lo + 1.0f) : uint64_t(0);
		};
		return axis(min_a.x, max_a.x, min_b.x, max_b.x) * axis(min_a.y, max_a.y, min_b.y, max_b.y) * axis(min_a.z, max_a.z, min_b.z, max_b.z);
	}

	uint64_t occupied_voxels(const BitVoxelMap& map)
	{
		const auto words = map.download_words();
		uint64_t n = 0;
		for (std::size_t v = 0; v < words.size() / 8; ++v)
		{
			BitVector256 bits{};
			for (int w = 0; w < 8; ++w)
				bits[w] = words[v * 8 + w];
			n += not_empty(bits) ? 1 : 0;
		}
		return n;
	}

	uint64_t occupied_voxels(const ProbVoxelMap& map, int minimum)
	{
		uint64_t n = 0;
		for (const int8_t v : map.download_occupancy())
			n += v >= minimum ? 1 : 0;
		return n;
	}

	/// eBVM_MAX_OCC_PROB / eBVM_MAX_FREE_PROB of CUDA (the swept volume meanings 254 and 4: occupancy updates +125 and -125).
	constexpr uint8_t kMaxOccProb = eBVM_SWEPT_VOLUME_END, kMaxFreeProb = eBVM_SWEPT_VOLUME_START;

	/// CUDA geometry_generation::Range: start, start + spacing, ... (float additions) while the value is <= end.
	std::vector<float> range_values(float lo, float hi, float spacing)
	{
		std::vector<float> values;
		for (float v = lo; v <= hi; v += spacing)
			values.push_back(v);
		return values;
	}

	/// CUDA createBoxOfPoints(min, max, delta) (x outermost, z innermost); the helper is not part of the Vulkan library.
	std::vector<Point3f> box_of_points(Point3f lo, Point3f hi, float delta)
	{
		std::vector<Point3f> points;
		for (const float x : range_values(lo.x, hi.x, delta))
			for (const float y : range_values(lo.y, hi.y, delta))
				for (const float z : range_values(lo.z, hi.z, delta))
					points.push_back({ x, y, z });
		return points;
	}

	/// CUDA createSphereOfPoints(center, radius, delta): the grid points of the bounding box that lie within the radius.
	std::vector<Point3f> sphere_of_points(Point3f c, float radius, float delta)
	{
		std::vector<Point3f> points;
		for (const float x : range_values(c.x - radius, c.x + radius, delta))
			for (const float y : range_values(c.y - radius, c.y + radius, delta))
				for (const float z : range_values(c.z - radius, c.z + radius, delta))
				{
					const float dx = c.x - x, dy = c.y - y, dz = c.z - z;
					if (std::sqrt(dx * dx + dy * dy + dz * dz) <= radius)
						points.push_back({ x, y, z });
				}
		return points;
	}

	void shift(std::vector<Point3f>& points, float dx, float dy, float dz)
	{
		for (Point3f& p : points)
		{
			p.x += dx;
			p.y += dy;
			p.z += dz;
		}
	}

	void expect_prob_equal(const ProbVoxelMap& map, const gvl_reference::ProbVoxelMapReference& ref, const std::string& what)
	{
		const auto gpu = map.download_occupancy();
		const Dim3ui d = map.dimensions();
		uint64_t different = 0, first = 0;
		for (uint64_t i = 0; i < gpu.size(); ++i)
			if (static_cast<int>(gpu[i]) != ref.get(i) && different++ == 0)
				first = i;
		EXPECT_EQ(different, 0u) << what << ": first difference at voxel (" << first % d.x << ", " << (first / d.x) % d.y << ", " << first / d.x / d.y
			<< "): " << (different == 0 ? 0 : int(gpu[first])) << " instead of " << ref.get(first);
	}

	class ExampleTest : public VulkanTest
	{
	protected:
		/// Calls `collide` on the list and compares it with the model of the state before the call.
		template<class Map, class Call>
		void expect_collision(BitVoxelList& list, const Map& map, Mode mode, Offset3i offset, float threshold, const BitVector256& mask, const Call& call,
			const std::string& what, uint32_t* count_out = nullptr)
		{
			const Expected expected = model(entries_of(list), view_of(map), mode, offset, threshold, mask);
			const auto [count, types] = call();
			EXPECT_EQ(count, expected.count) << what;
			if (mode == Mode::Types)
				EXPECT_EQ(types, expected.types) << what;
			EXPECT_EQ(list.download_bit_vectors(), expected.bits_after) << what << ": the colliding entries are marked";
			if (count_out != nullptr)
				*count_out = expected.count;
		}
	};
	using Result = std::pair<uint32_t, BitVector256>;
}

// Maps.cpp: one map of each type, a box, a point cloud file, a moving box that is cleared again (4 frames instead of the endless loop).
TEST_F(ExampleTest, Maps)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 200, 200, 200 }, 0.01f);
	EXPECT_NE(gvl.add_map(MT_BITVECTOR_VOXELMAP, "myBitmapVoxmap"), nullptr);
	EXPECT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "myBitmapVoxlist"), nullptr);
	EXPECT_NE(gvl.add_map(MT_BITVECTOR_OCTREE, "myBitmapOctree"), nullptr);
	EXPECT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "myProbabVoxmap"), nullptr);
	EXPECT_NE(gvl.add_map(MT_PROBAB_OCTREE, "myProbabOctree"), nullptr);

	const Point3f box_min{ 0.5f, 0.5f, 0.5f }, box_max{ 0.6f, 0.6f, 0.6f };
	EXPECT_TRUE(gvl.insert_box_into_map(box_min, box_max, "myBitmapVoxmap", eBVM_OCCUPIED));
	EXPECT_EQ(occupied_voxels(*gvl.get_map("myBitmapVoxmap")->as<BitVoxelMap>()), box_voxels(box_min, box_max, 0.01f));
	EXPECT_EQ(box_voxels(box_min, box_max, 0.01f), 12u * 12u * 12u) << "0.6 / 0.01f lies just above 60: the box has one voxel more per axis";

	EXPECT_TRUE(gvl.insert_point_cloud_from_file("myProbabVoxmap", "coordinate_system_100.binvox", kModels, eBVM_OCCUPIED, true, { 0, 0, 0 }, 0.5f));
	EXPECT_GT(occupied_voxels(*gvl.get_map("myProbabVoxmap")->as<ProbVoxelMap>(), 1), 100u);
	EXPECT_FALSE(gvl.visualize_map("myBitmapVoxmap"));
	EXPECT_FALSE(gvl.visualize_map("myProbabVoxmap"));
	EXPECT_TRUE(gvl.clear_map("myBitmapVoxmap"));
	EXPECT_EQ(occupied_voxels(*gvl.get_map("myBitmapVoxmap")->as<BitVoxelMap>()), 0u);

	float t = 0.0f;
	for (int frame = 0; frame < 4; ++frame)
	{
		t += 0.03f;
		const float x = std::sin(t), y = std::cos(t);
		const Point3f lo{ box_min.x + 0.2f * x, box_min.y + 0.2f * y, box_min.z }, hi{ box_max.x + 0.2f * x, box_max.y + 0.2f * y, box_max.z };
		EXPECT_TRUE(gvl.insert_box_into_map(lo, hi, "myBitmapVoxmap", eBVM_OCCUPIED, 2));
		EXPECT_EQ(occupied_voxels(*gvl.get_map("myBitmapVoxmap")->as<BitVoxelMap>()), box_voxels(lo, hi, 0.01f)) << "frame " << frame;
		EXPECT_TRUE(gvl.clear_map("myBitmapVoxmap"));
		EXPECT_EQ(occupied_voxels(*gvl.get_map("myBitmapVoxmap")->as<BitVoxelMap>()), 0u);
	}
}

// PrimitiveArrayTest.cpp: a list with four points and four primitive arrays, one per modifyPrimitives form.
TEST_F(ExampleTest, PrimitiveArrayTest)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 200, 200, 100 }, 3.0f);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "voxellist"), nullptr);
	const std::vector<Point3f> list_points{ { 30.0f, 9.0f, 12.0f }, { 30.0f, 15.0f, 12.0f }, { 30.0f, 21.0f, 12.0f }, { 30.0f, 27.0f, 12.0f } };
	EXPECT_TRUE(gvl.insert_point_cloud_into_map(list_points, "voxellist", 1));
	BitVoxelList& list = *gvl.get_map("voxellist")->as<BitVoxelList>();
	const auto coordinates = list.download_coordinates();
	ASSERT_EQ(coordinates.size(), 4u);
	for (std::size_t i = 0; i < 4; ++i) // x = 30 / 3, y = 3, 5, 7, 9, z = 12 / 3, all exact in float
		EXPECT_EQ((std::array<uint32_t, 3>{ coordinates[i].x, coordinates[i].y, coordinates[i].z }), (std::array<uint32_t, 3>{ 10, 3u + 2u * uint32_t(i), 4 }));

	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "metric3Sphere"));
	EXPECT_TRUE(gvl.modify_primitives("metric3Sphere", std::vector<Point3f>{ { 30.0f, 9.0f, 9.0f } }, 3.0f));
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "metric4Sphere"));
	EXPECT_TRUE(gvl.modify_primitives("metric4Sphere", std::vector<Point4f>{ { 30.0f, 15.0f, 9.0f, 3.0f } }));
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "voxel3Sphere"));
	EXPECT_TRUE(gvl.modify_primitives("voxel3Sphere", std::vector<Coord3i>{ { 10, 7, 3 } }, 1));
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "voxel4Sphere"));
	EXPECT_TRUE(gvl.modify_primitives("voxel4Sphere", std::vector<Coord4i>{ { 10, 9, 3, 1 } }));
	const std::map<std::string, PrimitiveEntry> expected{ { "metric3Sphere", { 10.5f, 3.5f, 3.5f, 0.5f } }, { "metric4Sphere", { 10.5f, 5.5f, 3.5f, 0.5f } },
		{ "voxel3Sphere", { 10.5f, 7.5f, 3.5f, 0.5f } }, { "voxel4Sphere", { 10.5f, 9.5f, 3.5f, 0.5f } } };
	for (const auto& [name, entry] : expected)
	{
		ASSERT_EQ(gvl.get_primitives(name)->size(), 1u) << name;
		EXPECT_EQ(gvl.get_primitives(name)->primitives()[0], entry) << name;
		EXPECT_FALSE(gvl.visualize_map("voxellist"));
		EXPECT_FALSE(gvl.visualize_primitives_array(name));
	}
	// The spheres hover above the list points: 3 m below them, in the same x column, y of the list entries 1 .. 3 (hand-checked from the example comment).
	EXPECT_EQ(gvl.get_primitives("metric3Sphere")->primitives()[0].y, 3.5f);
}

// Sandbox.cpp: three maps and the coordinate system from a file, boxes that move every frame, three collision calls and two primitive arrays
// of 1000 entries. Differences: the program uses corner3 before it is set in the first frame (undefined values), here the corners are calculated
// first; the program steps t by 0.03 and its boxes touch at t = 0 .. 0.52 (octree with bit map) and t = 1.05 .. 2.09 (probabilistic map with bit map),
// so the test uses the frames t = 0, 0.03, 1.5 and 1.53 instead of running for 50 frames.
TEST_F(ExampleTest, Sandbox)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 200, 200, 200 }, 0.01f);
	ASSERT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "myProbabVoxmap"), nullptr);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELMAP, "myBitmapVoxmap"), nullptr);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_OCTREE, "myOctree"), nullptr);
	ASSERT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "myCoordinateSystemMap"), nullptr);
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "myPrims"));
	EXPECT_TRUE(gvl.add_primitives(ePRIM_CUBOID, "mySecondPrims"));
	EXPECT_TRUE(gvl.insert_point_cloud_from_file("myCoordinateSystemMap", "coordinate_system_100.binvox", kModels, eBVM_OCCUPIED, true, { 0, 0, 0 }, 0.5f));

	const Point3f center_min{ 0.5f, 0.5f, 0.5f }, center_max{ 0.6f, 0.6f, 0.6f };
	int j = 0;
	uint32_t seen_prob_bit = 0, seen_octree = 0;
	std::vector<Point4f> prim_positions(1000);
	std::vector<Coord4i> prim_positions2(1000);
	for (const float t : { 0.0f, 0.03f, 1.5f, 1.53f })
	{
		const int frame = j / 1000;
		const float x = std::sin(t), y = std::cos(t);
		const auto moved = [&](float dx, float dy, float dz) { return std::make_pair(Point3f{ center_min.x + dx, center_min.y + dy, center_min.z + dz }, Point3f{ center_max.x + dx, center_max.y + dy, center_max.z + dz }); };
		const auto box1 = moved(0.2f * x, 0.2f * y, 0.0f), box2 = moved(0.0f, 0.2f * x, 0.2f * y), box3 = moved(0.2f * x, 0.0f, 0.2f * y);
		EXPECT_TRUE(gvl.insert_box_into_map(box1.first, box1.second, "myProbabVoxmap", eBVM_OCCUPIED, 2));
		EXPECT_TRUE(gvl.insert_box_into_map(box3.first, box3.second, "myBitmapVoxmap", eBVM_OCCUPIED, 2));
		EXPECT_TRUE(gvl.insert_box_into_map(box2.first, box2.second, "myOctree", eBVM_OCCUPIED, 2));

		ProbVoxelMap& prob = *gvl.get_map("myProbabVoxmap")->as<ProbVoxelMap>();
		BitVoxelMap& bit = *gvl.get_map("myBitmapVoxmap")->as<BitVoxelMap>();
		Octree& octree = *gvl.get_map("myOctree")->as<Octree>();
		EXPECT_EQ(occupied_voxels(bit), box_voxels(box3.first, box3.second, 0.01f));

		// myProbabVoxmap + myBitmapVoxmap (collideWith with the default threshold 1.0 = occupancy 127; one update gives 72, so use two thresholds)
		for (const float threshold : { 1.0f, 0.5f })
		{
			const auto prob_view = view_of(prob);
			const int level = level_of(threshold);
			const auto bit_view = view_of(bit);
			uint32_t expected = 0;
			for (uint32_t z = 0; z < 200; ++z)
				for (uint32_t yy = 0; yy < 200; ++yy)
					for (uint32_t xx = 0; xx < 200; ++xx)
						expected += (prob_view.occupancy_at({ xx, yy, z }) >= level && bit_view.bit_occupied({ xx, yy, z })) ? 1 : 0;
			EXPECT_EQ(prob.collide_with(bit, threshold), expected) << "frame " << frame << ", threshold " << threshold;
			seen_prob_bit += expected;
		}
		// myOctree + myBitmapVoxmap: both hold one box, so the collisions are the voxels of the intersection of the two boxes (voxel ranges by hand)
		const uint32_t octree_collisions = octree.collide_with(bit).collisions;
		EXPECT_EQ(octree_collisions, box_overlap(box2.first, box2.second, box3.first, box3.second, 0.01f)) << "frame " << frame;
		seen_octree += octree_collisions;
		EXPECT_EQ(octree.collide_with(bit).collisions, octree_collisions) << "repeatable";

		// the two primitive arrays: positions as in the program
		for (std::size_t i = 0; i < prim_positions.size(); ++i)
		{
			prim_positions[i] = { 0.2f + static_cast<float>(i / 250.0), 0.2f + static_cast<float>(std::sin(i / 5.0) / 50.0), static_cast<float>(std::sin(j / 5.0) / 50.0), 0.01f };
			prim_positions2[i] = { static_cast<int32_t>(20 + (std::sin(i / 5.0) / 0.5)), static_cast<int32_t>(20 + (std::sin(j / 5.0) / 0.5)), static_cast<int32_t>(i / 2.5), 1 };
			++j;
		}
		EXPECT_TRUE(gvl.modify_primitives("myPrims", prim_positions));
		EXPECT_TRUE(gvl.modify_primitives("mySecondPrims", prim_positions2));
		EXPECT_EQ(gvl.get_primitives("myPrims")->size(), 1000u);
		EXPECT_EQ(gvl.get_primitives("myPrims")->memory_usage_bytes(), 16000u);
		EXPECT_EQ(gvl.get_primitives("mySecondPrims")->size(), 1000u);
		for (const std::size_t i : { std::size_t(0), std::size_t(1), std::size_t(499), std::size_t(999) })
		{
			const PrimitiveEntry& metric = gvl.get_primitives("myPrims")->primitives()[i];
			EXPECT_EQ(metric.x, static_cast<float>(std::floor(double(prim_positions[i].x) / double(0.01f)) + 0.5)) << i;
			EXPECT_FLOAT_EQ(metric.radius, 0.5f);
			const PrimitiveEntry& voxel = gvl.get_primitives("mySecondPrims")->primitives()[i];
			EXPECT_EQ(voxel.x, static_cast<float>(prim_positions2[i].x) + 0.5f) << i;
			EXPECT_EQ(voxel.z, static_cast<float>(prim_positions2[i].z) + 0.5f) << i;
			EXPECT_EQ(voxel.radius, 0.5f);
		}
		EXPECT_FALSE(gvl.visualize_primitives_array("myPrims"));
		EXPECT_FALSE(gvl.visualize_primitives_array("mySecondPrims"));

		EXPECT_TRUE(gvl.clear_map("myProbabVoxmap"));
		EXPECT_TRUE(gvl.clear_map("myBitmapVoxmap"));
		EXPECT_TRUE(gvl.clear_map("myOctree"));
		EXPECT_EQ(occupied_voxels(bit), 0u);
		EXPECT_EQ(occupied_voxels(prob, -127), 0u) << "cleared maps hold no known voxel";
	}
	EXPECT_GT(seen_prob_bit, 0u) << "the boxes of the program overlap";
	EXPECT_GT(seen_octree, 0u);
}

// Collisions.cpp: a point cloud from a file and 50 swept volume boxes in two lists, two boxes in a probabilistic map that move every frame, collideWith
// and collideWithTypes of the lists against the map (3 frames).
TEST_F(ExampleTest, Collisions)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 200, 200, 200 }, 0.01f);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "myPointcloud"), nullptr);
	ASSERT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "myObstacles"), nullptr);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "mySweptVolume"), nullptr);
	EXPECT_TRUE(gvl.insert_point_cloud_from_file("myPointcloud", "robot4cmRes.pcd", kModels, eBVM_OCCUPIED, true, { 0.3f, 0.2f, 0.0f }, 0.5f));
	for (int i = 0; i < 50; ++i)
		EXPECT_TRUE(gvl.insert_box_into_map({ 0.3f + 0.025f * i, 0.7f, 0.3f }, { 0.4f + 0.025f * i, 0.8f, 0.7f }, "mySweptVolume",
			static_cast<uint8_t>(eBVM_SWEPT_VOLUME_START + i)));
	BitVoxelList& cloud = *gvl.get_map("myPointcloud")->as<BitVoxelList>();
	BitVoxelList& swept = *gvl.get_map("mySweptVolume")->as<BitVoxelList>();
	ProbVoxelMap& obstacles = *gvl.get_map("myObstacles")->as<ProbVoxelMap>();
	EXPECT_GT(cloud.size(), 1000u);
	EXPECT_GT(swept.size(), 5000u);

	const Point3f center1_min{ 0.5f, 0.5f, 0.5f }, center1_max{ 0.6f, 0.6f, 0.6f }, center2_min{ 0.5f, 0.5f, 0.3f }, center2_max{ 0.6f, 0.6f, 0.4f };
	float t = 0.0f;
	uint32_t cloud_collisions = 0, swept_collisions = 0;
	for (int frame = 0; frame < 3; ++frame)
	{
		t += 0.03f;
		const float x = std::sin(t), y = std::cos(t);
		EXPECT_TRUE(gvl.insert_box_into_map({ center1_min.x + 0.2f * x, center1_min.y + 0.2f * y, center1_min.z }, { center1_max.x + 0.2f * x, center1_max.y + 0.2f * y, center1_max.z },
			"myObstacles", eBVM_OCCUPIED, 2));
		EXPECT_TRUE(gvl.insert_box_into_map({ center2_min.x, center2_min.y + 0.4f * x, center2_min.z }, { center2_max.x, center2_max.y + 0.4f * x, center2_max.z },
			"myObstacles", eBVM_OCCUPIED, 2));

		// collideWith(map) with the default threshold needs occupancy 127 and the boxes were inserted once: nothing collides; a lower threshold does
		for (const float threshold : { 1.0f, 0.5f })
		{
			uint32_t expected_count = 0;
			expect_collision(cloud, obstacles, Mode::Default, {}, threshold, {}, [&] { return Result{ cloud.collide_with(obstacles, threshold), {} }; },
				"cloud collideWith, threshold " + std::to_string(threshold), &expected_count);
			cloud_collisions += expected_count;
		}
		uint32_t expected_swept = 0;
		expect_collision(swept, obstacles, Mode::Types, {}, 1.0f, {}, [&] { const auto r = swept.collide_with_types(obstacles, 1.0f); return Result{ r.count, r.types }; },
			"swept volume collideWithTypes", &expected_swept);
		swept_collisions += expected_swept;
		if (frame == 0)
		{
			// the bits of the colliding sweep steps are the meanings 4 + i of the boxes that overlap the obstacles' x range 0.5 .. 0.6 (+- 0.2 sin(t))
			const BitVector256 types = swept.collide_with_types(obstacles, 1.0f).types;
			uint32_t sweep_bits = types[0] & ~0xFu;
			for (int w = 1; w < 8; ++w)
				sweep_bits |= types[w];
			EXPECT_NE(sweep_bits, 0u) << "the colliding entries carry the meanings 4 + i of the boxes they belong to";
		}
		EXPECT_FALSE(gvl.visualize_map("myObstacles"));
		EXPECT_TRUE(gvl.clear_map("myObstacles"));
	}
	EXPECT_GT(cloud_collisions, 0u);
	EXPECT_GT(swept_collisions, 0u);
}

// VoxelListSandbox.cpp and VoxelListCountingSandbox.cpp (their collision calls are the same; the second also adds a counting list that nothing uses):
// three boxes with the meanings 34, 63 and 102 in a list, one box in a probabilistic map, collideWithTypeMask without a mask, with bit 63 and an
// offset, and the other masks of the program's comments.
TEST_F(ExampleTest, VoxelListSandbox)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 100, 100, 100 }, 0.1f);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "myVoxelList"), nullptr);
	ASSERT_NE(gvl.add_map(MT_COUNTING_VOXELLIST, "myCountingVoxelList"), nullptr);
	ASSERT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "myProbVoxelMap"), nullptr);
	EXPECT_TRUE(gvl.insert_box_into_map({ 0.09f, 0.09f, 0.09f }, { 0.91f, 0.41f, 0.41f }, "myProbVoxelMap", eBVM_OCCUPIED, 1));
	EXPECT_TRUE(gvl.insert_box_into_map({ 0.09f, 0.29f, 0.09f }, { 0.31f, 0.61f, 0.41f }, "myVoxelList", 34, 1));
	EXPECT_TRUE(gvl.insert_box_into_map({ 0.29f, 0.29f, 0.09f }, { 0.61f, 0.61f, 0.41f }, "myVoxelList", 63, 1));
	EXPECT_TRUE(gvl.insert_box_into_map({ 0.59f, 0.29f, 0.09f }, { 0.91f, 0.61f, 0.41f }, "myVoxelList", 102, 1));
	BitVoxelList& list = *gvl.get_map("myVoxelList")->as<BitVoxelList>();
	ProbVoxelMap& prob = *gvl.get_map("myProbVoxelMap")->as<ProbVoxelMap>();
	EXPECT_EQ(gvl.get_map("myVoxelList")->dimensions(), (Dim3ui{ 100, 100, 100 })) << "the façade reports the reference map for lists; the program prints getDimensions().x, the entries are size()";

	// Voxel counts by hand (all lengths in units of 0.1: lists 0.09 .. 0.31 = voxels 0 .. 4 etc. from the double formula).
	const uint64_t box34 = box_voxels({ 0.09f, 0.29f, 0.09f }, { 0.31f, 0.61f, 0.41f }, 0.1f), box63 = box_voxels({ 0.29f, 0.29f, 0.09f }, { 0.61f, 0.61f, 0.41f }, 0.1f),
		box102 = box_voxels({ 0.59f, 0.29f, 0.09f }, { 0.91f, 0.61f, 0.41f }, 0.1f);
	EXPECT_GT(list.size(), 0u);
	EXPECT_LE(list.size(), box34 + box63 + box102) << "the boxes overlap in one voxel layer, those entries are merged";
	EXPECT_GT(list.size(), std::max({ box34, box63, box102 }));

	// The program's calls: threshold 1.0 is raw here (occupancy >= 1), the prob. box has occupancy 72.
	BitVector256 mask{};
	uint32_t expected = 0;
	expect_collision(list, prob, Mode::Mask, {}, 1.0f, mask, [&] { return Result{ list.collide_with_type_mask(prob, mask, 1.0f), {} }; }, "no bits in the mask", &expected);
	EXPECT_EQ(expected, 0u);
	mask = bits_of({ 63 });
	expect_collision(list, prob, Mode::Mask, { 0, -1, 0 }, 1.0f, mask, [&] { return Result{ list.collide_with_type_mask(prob, mask, 1.0f, { 0, -1, 0 }), {} }; },
		"bit 63 with the offset 0,-1,0", &expected);
	EXPECT_GT(expected, 0u) << "box 63 (y voxels 2 .. 6) moved one voxel down meets the probabilistic box (y voxels 0 .. 4)";
	for (const int bit : { 34, 66, 102, 63 })
	{
		mask = bits_of({ bit });
		expect_collision(list, prob, Mode::Mask, {}, 1.0f, mask, [&] { return Result{ list.collide_with_type_mask(prob, mask, 1.0f), {} }; }, "bit " + std::to_string(bit), &expected);
		if (bit == 66)
			EXPECT_EQ(expected, 0u) << "no entry carries bit 66";
		else
			EXPECT_GT(expected, 0u) << "bit " << bit;
	}
	EXPECT_FALSE(gvl.visualize_map("myVoxelList"));
	EXPECT_FALSE(gvl.visualize_map("myProbVoxelMap"));
}

// CountingVoxelList.cpp: the robot cloud in a counting and a bit list, an obstacle box subtracted from both, three counting lists with box and points
// filtered with remove_underpopulated(0 / 1 / 2), the counting list collided with the bit list (thresholds 0 .. 2 instead of keyboard input).
// Difference: subtractFromCountingVoxelList / subtract take a metric offset in CUDA (the program passes zero), here there is none.
TEST_F(ExampleTest, CountingVoxelList)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 200, 200, 200 }, 0.01f);
	for (const char* name : { "CountingMap", "CountingFilter0Map", "CountingFilter1Map", "CountingFilter2Map" })
		ASSERT_NE(gvl.add_map(MT_COUNTING_VOXELLIST, name), nullptr);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "BitvectorMap"), nullptr);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "ObstacleBitvectorMap"), nullptr);
	EXPECT_TRUE(gvl.insert_point_cloud_from_file("BitvectorMap", "robot4cmRes.pcd", kModels, eBVM_OCCUPIED, true, { 0.3f, 0.2f, 0.0f }, 0.5f));
	EXPECT_TRUE(gvl.insert_point_cloud_from_file("CountingMap", "robot4cmRes.pcd", kModels, eBVM_OCCUPIED, true, { 0.3f, 0.2f, 0.0f }, 0.5f));
	EXPECT_TRUE(gvl.insert_box_into_map({ 0.0f, 0.0f, 0.2f }, { 2.0f, 2.0f, 0.5f }, "ObstacleBitvectorMap", eBVM_OCCUPIED, 1));

	CountingVoxelList& counting = *gvl.get_map("CountingMap")->as<CountingVoxelList>();
	BitVoxelList& bits = *gvl.get_map("BitvectorMap")->as<BitVoxelList>();
	BitVoxelList& obstacle = *gvl.get_map("ObstacleBitvectorMap")->as<BitVoxelList>();
	const auto counting_ids = counting.download_ids();
	const auto bit_ids = bits.download_ids();
	ASSERT_EQ(counting_ids, bit_ids) << "the same file gives the same voxels";
	ASSERT_GT(counting_ids.size(), 500u);
	const auto obstacle_ids = obstacle.download_ids();
	const std::set<uint32_t> removed(obstacle_ids.begin(), obstacle_ids.end());
	std::vector<uint32_t> remaining;
	for (const uint32_t id : counting_ids)
		if (!removed.contains(id))
			remaining.push_back(id);
	ASSERT_LT(remaining.size(), counting_ids.size()) << "the obstacle box cuts something off the robot";
	ASSERT_GT(remaining.size(), 0u);

	counting.subtract_from_counting_list(obstacle);
	bits.subtract(obstacle);
	EXPECT_EQ(counting.download_ids(), remaining);
	EXPECT_EQ(bits.download_ids(), remaining);

	const Point3f filter_min{ 0.1f, 0.5f, 0.7f }, filter_max{ 0.5f, 1.0f, 0.9f };
	const char* const filters[] = { "CountingFilter0Map", "CountingFilter1Map", "CountingFilter2Map" };
	for (const char* name : filters)
		EXPECT_TRUE(gvl.insert_box_into_map(filter_min, filter_max, name, eBVM_OCCUPIED, 1));
	const uint64_t box = box_voxels(filter_min, filter_max, 0.01f);
	const std::vector<Point3f> points{ { 0.2f, 0.7f, 0.5f }, { 0.2f, 0.7f, 0.4f }, { 0.2f, 0.7f, 0.4f }, { 0.2f, 0.7f, 0.3f }, { 0.2f, 0.7f, 0.3f }, { 0.2f, 0.7f, 0.3f } };
	for (const char* name : filters)
		EXPECT_TRUE(gvl.insert_point_cloud_into_map(points, name, eBVM_OCCUPIED));
	std::size_t index = 0;
	for (const char* name : filters)
	{
		CountingVoxelList& filtered = *gvl.get_map(name)->as<CountingVoxelList>();
		EXPECT_EQ(filtered.size(), box + 3) << "box voxels and three point voxels (z = 0.5, 0.4, 0.3 are below the box)";
		filtered.remove_underpopulated(static_cast<int8_t>(index));
		// counts: the box voxels 1, the points 1, 2 and 3 -> threshold 2 keeps the last two
		EXPECT_EQ(filtered.size(), index == 2 ? 2u : box + 3) << name;
		++index;
	}
	EXPECT_EQ(gvl.get_map("CountingFilter2Map")->as<CountingVoxelList>()->download_counts(), (std::vector<int8_t>{ 3, 2 }))
		<< "ids ascend with z, so the z = 0.3 voxel (count 3) comes before the z = 0.4 voxel (count 2)";

	// collideWith(bit list, threshold): every entry of CountingMap has its id in the bit list, so the collisions are the entries with count >= threshold
	const auto counts = counting.download_counts();
	for (const int threshold : { 0, 1, 2, 3 })
	{
		const auto expected = std::count_if(counts.begin(), counts.end(), [threshold](int8_t c) { return c >= threshold; });
		EXPECT_EQ(counting.collide_with(bits, static_cast<int8_t>(threshold)), static_cast<uint32_t>(expected)) << "threshold " << threshold;
	}
	EXPECT_EQ(counting.collide_with(bits, 0), remaining.size());
}

// ProbVoxels.cpp: shifted boxes with swept-volume meanings (occupancy updates meaning - 129) and a slab of "uncertain" points. The program's
// createBoxOfPoints is re-implemented above. Expectations: the sparse reference (sum of the updates per voxel, clamped) and hand-calculated voxels.
TEST_F(ExampleTest, ProbVoxels)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 136, 136, 136 }, 1.0f);
	const GpuVoxelsMapSharedPtr map = gvl.add_map(MT_PROBAB_VOXELMAP, "myProbVoxelMap");
	ASSERT_NE(map, nullptr);
	const ProbVoxelMap& prob = *map->as<ProbVoxelMap>();
	gvl_reference::ProbVoxelMapReference ref(136, 136, 136, 1.0f);

	const auto insert = [&](const std::vector<Point3f>& points, uint8_t meaning)
	{
		const uint32_t outside = map->insert_point_cloud(points, meaning);
		const auto events = ref.point_events(points, static_cast<int>(meaning) - 129);
		ref.apply_accumulated(events.events);
		EXPECT_EQ(outside, events.outside) << "meaning " << int(meaning);
		EXPECT_EQ(outside, 0u) << "all boxes lie inside the map (meaning " << int(meaning) << ")";
	};

	std::vector<Point3f> box = box_of_points({ 10, 10, 30 }, { 30, 30, 50 }, 0.9f);
	EXPECT_EQ(box.size(), 23u * 23u * 23u) << "10 + 0.9 k <= 30 for k = 0 .. 22";
	insert(box, kMaxOccProb);
	shift(box, 0, 15, 15);
	insert(box, kMaxOccProb);
	for (const uint8_t meaning : { 229, 204, 179, 154 })
	{
		shift(box, 0, 15, 15);
		insert(box, meaning);
	}
	expect_prob_equal(prob, ref, "after the occupied boxes");
	shift(box, 0, 15, -15);
	insert(box, 104);
	for (const uint8_t meaning : { 79, 54, 29, 4, 4 })
	{
		shift(box, 0, -15, -15);
		insert(box, meaning);
	}

	insert(box_of_points({ 10, 5, 5 }, { 12, 130, 130 }, 0.9f), eBVM_UNCERTAIN_OCC_PROB);
	expect_prob_equal(prob, ref, "ProbVoxels");

	// Hand-calculated voxels (points of the first box per axis: 10, 10.9, 11.8, 12.7, 13.6, ...; of the slab: x in {10, 10.9, 11.8}).
	const Dim3ui dims = prob.dimensions();
	const auto voxel = [&](uint32_t x, uint32_t y, uint32_t z) { return int(prob.download_occupancy(ProbVoxelMap::voxel_index(dims, { x, y, z }), 1)[0]); };
	EXPECT_EQ(voxel(10, 10, 30), 127) << "8 points of +125 saturate; the slab adds 0";
	EXPECT_EQ(voxel(13, 11, 31), 125) << "one point (13.6, 11.8, 31.8) of +125";
	EXPECT_EQ(voxel(11, 5, 5), 0) << "only the slab touches it: unknown plus 0 is uncertain, not unknown";
	EXPECT_EQ(voxel(0, 0, 0), -128);
	EXPECT_EQ(voxel(135, 135, 135), -128);
	const auto occupancy = prob.download_occupancy();
	EXPECT_NE(std::find(occupancy.begin(), occupancy.end(), int8_t(-125)), occupancy.end()) << "voxels with a single point of the free boxes";
}

// ProbMap2DistMap.cpp / ProbMap2DistMapExample.cpp (the two programs do the same): two overlapping boxes as occupied / free space, merge_occupied
// with threshold 0.55, then parallel banding with m1 = 1, m2 = 1, m3 = 4. The comment of the program ("the overlap has probability 0.5") does not
// hold with spacing 0.9: every voxel of a box holds at least one point, so +127 minus 125 per point leaves at most 2.
TEST_F(ExampleTest, ProbMap2DistMap)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 128, 128, 128 }, 1.0f);
	const GpuVoxelsMapSharedPtr prob_map = gvl.add_map(MT_PROBAB_VOXELMAP, "myProbVoxelMap");
	const GpuVoxelsMapSharedPtr dist_map = gvl.add_map(MT_DISTANCE_VOXELMAP, "myDistanceMap");
	ASSERT_NE(prob_map, nullptr);
	ASSERT_NE(dist_map, nullptr);
	ProbVoxelMap& prob = *prob_map->as<ProbVoxelMap>();
	DistanceVoxelMap& dist = *dist_map->as<DistanceVoxelMap>();
	gvl_reference::ProbVoxelMapReference ref(128, 128, 128, 1.0f);

	const auto points1 = box_of_points({ 40, 40, 40 }, { 60, 60, 60 }, 0.9f);
	const auto points2 = box_of_points({ 50, 50, 50 }, { 70, 70, 70 }, 0.9f);
	EXPECT_EQ(prob_map->insert_point_cloud(points1, kMaxOccProb), 0u);
	EXPECT_EQ(prob_map->insert_point_cloud(points2, kMaxFreeProb), 0u);
	ref.apply_accumulated(ref.point_events(points1, 125).events);
	ref.apply_accumulated(ref.point_events(points2, -125).events);
	expect_prob_equal(prob, ref, "ProbMap2DistMap prob map");

	dist.merge_occupied(prob, 0.55f);
	dist.compute_distances(DistanceAlgorithm::ParallelBanding, { 1, 1, 4 });

	// Hand count: both boxes cover the voxels 40 .. 59 and 50 .. 69 per axis (every voxel holds a point, the next point after 59.8 is 60.7 > 60).
	// The first box is occupied (>= 125) except the 10^3 voxels it shares with the second one (<= 2, threshold: > 12).
	const int level = gvl_reference::ProbVoxelMapReference::probability_from_float(0.55f);
	EXPECT_EQ(level, 12);
	const Dim3ui dims = dist.dimensions();
	std::vector<std::array<int, 3>> obstacles;
	std::vector<bool> is_obstacle(dims.voxel_count());
	for (uint32_t z = 0; z < dims.z; ++z)
		for (uint32_t y = 0; y < dims.y; ++y)
			for (uint32_t x = 0; x < dims.x; ++x)
			{
				const uint64_t index = DistanceVoxelMap::voxel_index(dims, { x, y, z });
				is_obstacle[index] = ref.get(index) > level;
				if (is_obstacle[index])
					obstacles.push_back({ int(x), int(y), int(z) });
			}
	EXPECT_EQ(obstacles.size(), 20u * 20u * 20u - 10u * 10u * 10u);

	const auto nearest = dist.download_obstacles();
	uint64_t wrong_obstacles = 0;
	for (uint32_t z = 0; z < dims.z; ++z)
		for (uint32_t y = 0; y < dims.y; ++y)
			for (uint32_t x = 0; x < dims.x; ++x)
			{
				const uint64_t index = DistanceVoxelMap::voxel_index(dims, { x, y, z });
				const Coord3ui n = nearest[index];
				wrong_obstacles += ((n.x == x && n.y == y && n.z == z) != is_obstacle[index]) ? 1 : 0;
			}
	EXPECT_EQ(wrong_obstacles, 0u) << "a voxel stores itself exactly when it is an obstacle";

	// Brute force squared distances of a sample, and hand-calculated ones.
	std::vector<uint32_t> indices;
	for (uint64_t i = 0; i < dims.voxel_count(); i += 997)
		indices.push_back(static_cast<uint32_t>(i));
	const auto distances = dist.squared_distances(indices);
	uint64_t wrong_distances = 0;
	for (std::size_t k = 0; k < indices.size(); ++k)
	{
		const int x = indices[k] % dims.x, y = (indices[k] / dims.x) % dims.y, z = indices[k] / dims.x / dims.y;
		int64_t best = std::numeric_limits<int64_t>::max();
		for (const auto& o : obstacles)
			best = std::min<int64_t>(best, int64_t(o[0] - x) * (o[0] - x) + int64_t(o[1] - y) * (o[1] - y) + int64_t(o[2] - z) * (o[2] - z));
		wrong_distances += distances[k] != best ? 1 : 0;
	}
	EXPECT_EQ(wrong_distances, 0u) << indices.size() << " sampled voxels";
	EXPECT_EQ(dist.squared_obstacle_distance({ 45, 45, 45 }), 0);
	EXPECT_EQ(dist.squared_obstacle_distance({ 30, 30, 30 }), 300) << "to (40, 40, 40)";
	EXPECT_EQ(dist.squared_obstacle_distance({ 50, 50, 50 }), 1) << "inside the shared part, next to (49, 50, 50)";
	EXPECT_EQ(dist.squared_obstacle_distance({ 60, 60, 60 }), 123) << "to (59, 59, 49): 1 + 1 + 11^2";
	EXPECT_EQ(dist.squared_obstacle_distance({ 127, 127, 127 }), 78 * 78 + 68 * 68 + 68 * 68) << "to (59, 59, 49)";
}

// RaycastingSensorExample.cpp: two probabilistic maps, one with the robot cut out, a sensor that moves along a line (5 of its 50 positions).
// Sphere points are generated as in CUDA; expectations: the sparse reference with the ray walk translated from the CUDA code (Accumulate mode:
// the updates of one call are summed, the CUDA program applies them in an unspecified order).
TEST_F(ExampleTest, RaycastingSensorExample)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 128, 128, 128 }, 1.0f);
	const GpuVoxelsMapSharedPtr map1 = gvl.add_map(MT_PROBAB_VOXELMAP, "myProbVoxelMap");
	const GpuVoxelsMapSharedPtr map2 = gvl.add_map(MT_PROBAB_VOXELMAP, "myProbVoxelMap2");
	const GpuVoxelsMapSharedPtr robot_map = gvl.add_map(MT_BITVECTOR_VOXELMAP, "myRobotMap");
	ASSERT_NE(robot_map, nullptr);
	ProbVoxelMap& prob1 = *map1->as<ProbVoxelMap>();
	ProbVoxelMap& prob2 = *map2->as<ProbVoxelMap>();
	const BitVoxelMap& robot = *robot_map->as<BitVoxelMap>();

	std::vector<Point3f> obstacle = sphere_of_points({ 50, 65, 50 }, 10.0f, 0.9f);
	const std::size_t first_sphere = obstacle.size();
	const std::vector<Point3f> sphere2 = sphere_of_points({ 65, 50, 50 }, 8.0f, 0.9f);
	obstacle.insert(obstacle.end(), sphere2.begin(), sphere2.end());
	EXPECT_GT(first_sphere, 5000u);
	EXPECT_GT(sphere2.size(), 2000u);

	const auto robot1 = box_of_points({ 35, 50, 35 }, { 50, 90, 65 }, 0.9f);
	const auto robot2 = box_of_points({ 35, 65, 35 }, { 65, 80, 65 }, 0.9f);
	EXPECT_EQ(robot_map->insert_point_cloud(robot1, eBVM_OCCUPIED), 0u);
	EXPECT_EQ(robot_map->insert_point_cloud(robot2, eBVM_OCCUPIED), 0u);
	gvl_reference::BitVoxelMapReference robot_ref(128, 128, 128, 1.0f);
	robot_ref.insert_points(robot1, eBVM_OCCUPIED);
	robot_ref.insert_points(robot2, eBVM_OCCUPIED);

	gvl_reference::ProbVoxelMapReference ref1(128, 128, 128, 1.0f), ref2(128, 128, 128, 1.0f);
	const Dim3ui dims = prob1.dimensions();
	for (const std::size_t pos : { 50, 51, 52, 98, 99 })
	{
		const Point3f sensor{ float(pos), 2.0f, float(pos / 2.0) };
		const float origin[3] = { sensor.x, sensor.y, sensor.z };
		const auto cut = prob1.insert_sensor_data(obstacle, sensor, true, &robot, eBVM_OCCUPIED);
		const auto uncut = prob2.insert_sensor_data(obstacle, sensor, true, nullptr);

		const auto events1 = ref1.sensor_events(obstacle, origin, true, &robot_ref, eBVM_OCCUPIED, -10, 72);
		const auto events2 = ref2.sensor_events(obstacle, origin, true, nullptr, eBVM_OCCUPIED, -10, 72);
		ref1.apply_accumulated(events1.events);
		ref2.apply_accumulated(events2.events);
		EXPECT_EQ(cut.cut_by_robot, events1.cut_by_robot) << "pos " << pos;
		EXPECT_EQ(cut.outside, events1.outside) << "pos " << pos;
		EXPECT_EQ(uncut.cut_by_robot, 0u) << "pos " << pos << ": no robot map, nothing is cut";
		EXPECT_EQ(uncut.outside, events2.outside) << "pos " << pos;
		EXPECT_GT(cut.cut_by_robot, 1000u) << "pos " << pos << ": most of the first sphere lies inside the robot boxes";
		EXPECT_EQ(cut.outside, 0u);
		expect_prob_equal(prob1, ref1, "robot cut, pos " + std::to_string(pos));
		expect_prob_equal(prob2, ref2, "no cut, pos " + std::to_string(pos));

		// The sensor voxel is the start of every ray: thousands of free updates of -10 saturate it.
		const uint64_t sensor_voxel = ProbVoxelMap::voxel_index(dims, { uint32_t(pos), 2u, uint32_t(pos / 2) });
		EXPECT_EQ(int(prob1.download_occupancy(sensor_voxel, 1)[0]), -127) << "pos " << pos;
		EXPECT_EQ(int(prob2.download_occupancy(sensor_voxel, 1)[0]), -127) << "pos " << pos;
	}
	EXPECT_NE(prob1.download_occupancy(), prob2.download_occupancy()) << "the cut changes the first map";
}
