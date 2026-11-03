// Slice S7-5: headless versions of examples of packages/examples as smoke tests. Each test repeats the GpuVoxels call sequence of one example
// without the viewer and the endless loop (a few frames instead), and checks the results against numbers that do not come from the call under
// test: hand-calculated counts, the voxel formula in double precision, or host models built from the downloaded states of the maps and lists.
// Differences to the programs are noted at each test. Not translated (and why): see Agent/knowledge_base/architecture/vulkan_gpu_voxels_examples.md.
#include "vulkan_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>

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
