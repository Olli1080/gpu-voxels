// Slice S7-5: the host container of primitives (PrimitiveArray) and the primitives members of the façade. The expectations are computed in
// double precision from the formulas of primitive_array::PrimitiveArray::setPoints (voxel = floor(p / side) + 0.5, radius = (size / side) / 2),
// or are hand-calculated numbers of the PrimitiveArrayTest example.
#include "vulkan_fixture.h"

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>
#include <gpu_voxels/vulkan/PrimitiveArray.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;

namespace
{
	constexpr Dim3ui kDims{ 200, 200, 100 };

	void expect_error(const std::function<void()>& call, const std::string& text)
	{
		try
		{
			call();
			ADD_FAILURE() << "no exception, expected one with '" << text << "'";
		}
		catch (const VulkanError& e)
		{
			EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << "message was: " << e.what();
		}
	}

	std::vector<PrimitiveEntry> entries(const PrimitiveArray& array) { return { array.primitives().begin(), array.primitives().end() }; }

	PrimitiveEntry entry(float x, float y, float z, float radius) { return { x, y, z, radius }; }

	class PrimitiveTest : public VulkanTest
	{
	};
}

TEST_F(PrimitiveTest, MetricPositionsBecomeVoxelCentresAndHalfSizes)
{
	PrimitiveArray array(kDims, 0.5f, ePRIM_SPHERE);
	// 0.5 is exact in float: 1.3 / 0.5 = 2.6 -> 2, -0.2 / 0.5 = -0.4 -> -1, 3.9 / 0.5 = 7.8 -> 7; the size 3.0 / 0.5 / 2 = 3.
	const std::vector<Point4f> points{ { 1.3f, -0.2f, 3.9f, 3.0f }, { 1.0f, -0.5f, 0.0f, 0.5f }, { 0.0f, 0.49f, -0.01f, 0.0f } };
	array.set_points(points);
	EXPECT_EQ(entries(array), (std::vector<PrimitiveEntry>{ entry(2.5f, -0.5f, 7.5f, 3.0f), entry(2.5f, -0.5f, 0.5f, 0.5f), entry(0.5f, 0.5f, -0.5f, 0.0f) }))
		<< "exact multiples of the side length belong to the voxel that starts there";
	EXPECT_EQ(array.size(), 3u);
	EXPECT_EQ(array.memory_usage_bytes(), 3u * 16u);

	const std::vector<Point3f> same_size{ { 1.3f, -0.2f, 3.9f }, { 2.0f, 2.0f, 2.0f } };
	array.set_points(same_size, 2.0f);
	EXPECT_EQ(entries(array), (std::vector<PrimitiveEntry>{ entry(2.5f, -0.5f, 7.5f, 2.0f), entry(4.5f, 4.5f, 4.5f, 2.0f) }));

	// A size of zero is allowed (a point), only negative sizes are refused.
	array.set_points(same_size, 0.0f);
	EXPECT_EQ(entries(array), (std::vector<PrimitiveEntry>{ entry(2.5f, -0.5f, 7.5f, 0.0f), entry(4.5f, 4.5f, 4.5f, 0.0f) }));
}

TEST_F(PrimitiveTest, VoxelPositionsAreShiftedByHalfAndKeepHalfTheSize)
{
	PrimitiveArray array(kDims, 3.0f, ePRIM_CUBOID);
	array.set_points(std::vector<Coord4i>{ { 3, -2, 7, 5 }, { 0, 0, 0, 0 }, { -1, 100, 99, 1 } });
	EXPECT_EQ(entries(array), (std::vector<PrimitiveEntry>{ entry(3.5f, -1.5f, 7.5f, 2.5f), entry(0.5f, 0.5f, 0.5f, 0.0f), entry(-0.5f, 100.5f, 99.5f, 0.5f) }))
		<< "the voxel side length does not matter for voxel positions";
	array.set_points(std::vector<Coord3i>{ { 10, 7, 3 }, { -4, 0, 1 } }, 3);
	EXPECT_EQ(entries(array), (std::vector<PrimitiveEntry>{ entry(10.5f, 7.5f, 3.5f, 1.5f), entry(-3.5f, 0.5f, 1.5f, 1.5f) }));
	array.set_points(std::vector<Coord3i>{ { 1, 2, 3 } }, 0);
	EXPECT_EQ(entries(array), (std::vector<PrimitiveEntry>{ entry(1.5f, 2.5f, 3.5f, 0.0f) }));
	// Large voxel coordinates survive up to the precision of float (CUDA converts the same way).
	array.set_points(std::vector<Coord3i>{ { 1 << 20, -(1 << 20), 16777216 } }, 1u << 30);
	EXPECT_EQ(array.primitives()[0].x, 1048576.5f);
	EXPECT_EQ(array.primitives()[0].y, -1048575.5f);
	EXPECT_EQ(array.primitives()[0].radius, 536870912.0f);
}

TEST_F(PrimitiveTest, RandomMetricPositionsMatchTheFormulaInDoublePrecision)
{
	std::mt19937 rng(11);
	std::uniform_real_distribution<float> position(-5.0f, 20.0f);
	std::uniform_real_distribution<float> size(0.0f, 2.0f);
	for (const float side : { 0.1f, 0.25f, 3.0f })
	{
		PrimitiveArray array(kDims, side, ePRIM_SPHERE);
		std::vector<Point4f> four;
		std::vector<Point3f> three;
		for (int i = 0; i < 300; ++i)
		{
			four.push_back({ position(rng), position(rng), position(rng), size(rng) });
			three.push_back({ four.back().x, four.back().y, four.back().z });
		}
		array.set_points(four);
		ASSERT_EQ(array.size(), four.size());
		for (std::size_t i = 0; i < four.size(); ++i)
		{
			const PrimitiveEntry& e = array.primitives()[i];
			EXPECT_EQ(e.x, static_cast<float>(std::floor(double(four[i].x) / side) + 0.5)) << side << " " << i;
			EXPECT_EQ(e.y, static_cast<float>(std::floor(double(four[i].y) / side) + 0.5)) << side << " " << i;
			EXPECT_EQ(e.z, static_cast<float>(std::floor(double(four[i].z) / side) + 0.5)) << side << " " << i;
			EXPECT_FLOAT_EQ(e.radius, static_cast<float>(double(four[i].w) / side / 2.0)) << side << " " << i;
		}
		array.set_points(three, 0.7f);
		ASSERT_EQ(array.size(), three.size());
		for (std::size_t i = 0; i < three.size(); ++i)
		{
			const PrimitiveEntry& e = array.primitives()[i];
			EXPECT_EQ(e.x, static_cast<float>(std::floor(double(three[i].x) / side) + 0.5)) << side << " " << i;
			EXPECT_FLOAT_EQ(e.radius, static_cast<float>(double(0.7f) / side / 2.0)) << side << " " << i;
		}
	}
}

TEST_F(PrimitiveTest, SetPointsReplacesTheContentAndEmptyInputClearsIt)
{
	PrimitiveArray array(kDims, 1.0f, ePRIM_SPHERE);
	EXPECT_EQ(array.size(), 0u);
	EXPECT_EQ(array.memory_usage_bytes(), 0u);
	array.set_points(std::vector<Coord4i>{ { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 3, 3, 3, 3 } });
	ASSERT_EQ(array.size(), 3u);
	array.set_points(std::vector<Coord4i>{ { 9, 9, 9, 4 } });
	EXPECT_EQ(entries(array), (std::vector<PrimitiveEntry>{ entry(9.5f, 9.5f, 9.5f, 2.0f) }));
	EXPECT_EQ(array.memory_usage_bytes(), 16u);
	array.set_points(std::vector<Point4f>{});
	EXPECT_EQ(array.size(), 0u);
	array.set_points(std::vector<Coord4i>{ { 5, 5, 5, 2 } });
	array.set_points(std::vector<Coord3i>{}, 4);
	EXPECT_EQ(array.size(), 0u);
	EXPECT_EQ(array.type(), ePRIM_SPHERE);
	EXPECT_EQ(array.dimensions(), kDims);
	EXPECT_EQ(array.voxel_side_length(), 1.0f);
	EXPECT_EQ(PrimitiveArray(kDims, 1.0f, ePRIM_CUBOID).type(), ePRIM_CUBOID);
}

TEST_F(PrimitiveTest, InvalidInputIsRefusedAndChangesNothing)
{
	PrimitiveArray array(kDims, 0.5f, ePRIM_SPHERE);
	array.set_points(std::vector<Coord4i>{ { 1, 2, 3, 4 } });
	const auto before = entries(array);
	const float nan = std::numeric_limits<float>::quiet_NaN();
	const float inf = std::numeric_limits<float>::infinity();

	expect_error([&] { array.set_points(std::vector<Point4f>{ { 1, 1, 1, 1 }, { 1, 1, 1, -0.5f } }); }, "the size of primitive 1 is negative");
	expect_error([&] { array.set_points(std::vector<Coord4i>{ { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, -1 } }); }, "the size of primitive 2 is negative");
	expect_error([&] { array.set_points(std::vector<Point3f>{ { 1, 1, 1 } }, -1.0f); }, "the diameter is negative");
	expect_error([&] { array.set_points(std::vector<Point4f>{ { 1, 1, 1, 1 }, { nan, 1, 1, 1 } }); }, "primitive 1 is not finite in voxel units");
	expect_error([&] { array.set_points(std::vector<Point4f>{ { 1, inf, 1, 1 } }); }, "primitive 0 is not finite in voxel units");
	expect_error([&] { array.set_points(std::vector<Point4f>{ { 1, 1, -inf, 1 } }); }, "primitive 0 is not finite in voxel units");
	expect_error([&] { array.set_points(std::vector<Point4f>{ { 1, 1, 1, nan } }); }, "primitive 0 is not finite in voxel units");
	expect_error([&] { array.set_points(std::vector<Point4f>{ { 1, 1, 1, inf } }); }, "primitive 0 is not finite in voxel units");
	expect_error([&] { array.set_points(std::vector<Point4f>{ { 3.0e38f, 1, 1, 1 } }); }, "primitive 0 is not finite in voxel units"); // overflows after / 0.5
	expect_error([&] { array.set_points(std::vector<Point4f>{ { 1, 1, 1, 3.0e38f } }); }, "primitive 0 is not finite in voxel units");
	expect_error([&] { array.set_points(std::vector<Point3f>{ { 1, 1, nan } }, 1.0f); }, "primitive 0 is not finite in voxel units");
	expect_error([&] { array.set_points(std::vector<Point3f>{ { 1, 1, 1 } }, nan); }, "primitive 0 is not finite in voxel units");
	expect_error([&] { array.set_points(std::vector<Point3f>{ { 1, 1, 1 } }, inf); }, "primitive 0 is not finite in voxel units");
	EXPECT_EQ(entries(array), before);

	expect_error([&] { PrimitiveArray(kDims, 1.0f, static_cast<PrimitiveType>(2)); }, "unknown primitive type 2");
	expect_error([&] { PrimitiveArray(kDims, 1.0f, static_cast<PrimitiveType>(255)); }, "unknown primitive type 255");
	for (const float side : { 0.0f, -1.0f, nan, inf })
		expect_error([&] { PrimitiveArray(kDims, side, ePRIM_SPHERE); }, "the voxel side length must be positive and finite");
}

TEST_F(PrimitiveTest, TheFacadeKeepsNamedArraysLikeTheExample)
{
	GpuVoxels gvl(ctx());
	expect_error([&] { (void)gvl.add_primitives(ePRIM_SPHERE, "a"); }, "initialize() has to be called before primitive arrays are added");
	gvl.initialize(kDims, 3.0f);

	// PrimitiveArrayTest.cpp: 200 x 200 x 100 voxels of 3 m.
	const std::vector<Point3f> metric3{ { 30.0f, 9.0f, 9.0f } };
	const std::vector<Point4f> metric4{ { 30.0f, 15.0f, 9.0f, 3.0f } };
	const std::vector<Coord3i> voxel3{ { 10, 7, 3 } };
	const std::vector<Coord4i> voxel4{ { 10, 9, 3, 1 } };
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "metric3Sphere"));
	EXPECT_TRUE(gvl.modify_primitives("metric3Sphere", metric3, 3.0f));
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "metric4Sphere"));
	EXPECT_TRUE(gvl.modify_primitives("metric4Sphere", metric4));
	EXPECT_TRUE(gvl.add_primitives(ePRIM_CUBOID, "voxel3Cuboid"));
	EXPECT_TRUE(gvl.modify_primitives("voxel3Cuboid", voxel3, 1));
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "voxel4Sphere"));
	EXPECT_TRUE(gvl.modify_primitives("voxel4Sphere", voxel4));

	EXPECT_EQ(entries(*gvl.get_primitives("metric3Sphere")), (std::vector<PrimitiveEntry>{ entry(10.5f, 3.5f, 3.5f, 0.5f) }));
	EXPECT_EQ(entries(*gvl.get_primitives("metric4Sphere")), (std::vector<PrimitiveEntry>{ entry(10.5f, 5.5f, 3.5f, 0.5f) }));
	EXPECT_EQ(entries(*gvl.get_primitives("voxel3Cuboid")), (std::vector<PrimitiveEntry>{ entry(10.5f, 7.5f, 3.5f, 0.5f) }));
	EXPECT_EQ(entries(*gvl.get_primitives("voxel4Sphere")), (std::vector<PrimitiveEntry>{ entry(10.5f, 9.5f, 3.5f, 0.5f) }));
	EXPECT_EQ(gvl.get_primitives("metric3Sphere")->type(), ePRIM_SPHERE);
	EXPECT_EQ(gvl.get_primitives("voxel3Cuboid")->type(), ePRIM_CUBOID);
	EXPECT_EQ(gvl.get_primitives("voxel3Cuboid")->dimensions(), kDims);
	EXPECT_EQ(gvl.get_primitives("voxel3Cuboid")->voxel_side_length(), 3.0f);
	EXPECT_NE(gvl.get_primitives("metric3Sphere"), gvl.get_primitives("metric4Sphere"));
	EXPECT_EQ(gvl.get_primitives("metric3Sphere"), gvl.get_primitives("metric3Sphere"));

	// The viewer does not exist.
	EXPECT_FALSE(gvl.visualize_primitives_array("metric3Sphere"));
	EXPECT_FALSE(gvl.visualize_primitives_array("metric3Sphere", false));
	EXPECT_FALSE(gvl.visualize_primitives_array("nothing"));
}

TEST_F(PrimitiveTest, TheFacadeNamesAreSeparateAndUnknownNamesReturnFalse)
{
	GpuVoxels gvl(ctx());
	gvl.initialize({ 20, 20, 20 }, 0.5f);
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "x"));
	EXPECT_FALSE(gvl.add_primitives(ePRIM_CUBOID, "x")) << "a used name";
	EXPECT_EQ(gvl.get_primitives("x")->type(), ePRIM_SPHERE) << "the first array is still there";
	EXPECT_TRUE(gvl.add_map(MT_BITVECTOR_VOXELMAP, "x") != nullptr) << "map names are separate from array names";
	EXPECT_NE(gvl.get_map("x"), nullptr);
	expect_error([&] { (void)gvl.add_primitives(static_cast<PrimitiveType>(7), "bad"); }, "unknown primitive type 7");
	EXPECT_EQ(gvl.get_primitives("bad"), nullptr) << "a refused array is not added";

	const std::vector<Point4f> a{ { 1, 1, 1, 1 } };
	const std::vector<Coord4i> b{ { 1, 1, 1, 1 } };
	const std::vector<Point3f> c{ { 1, 1, 1 } };
	const std::vector<Coord3i> d{ { 1, 1, 1 } };
	EXPECT_FALSE(gvl.modify_primitives("nope", a));
	EXPECT_FALSE(gvl.modify_primitives("nope", b));
	EXPECT_FALSE(gvl.modify_primitives("nope", c, 1.0f));
	EXPECT_FALSE(gvl.modify_primitives("nope", d, 1));
	EXPECT_FALSE(gvl.del_primitives("nope"));
	EXPECT_EQ(gvl.get_primitives("nope"), nullptr);

	// Each of the four forms changes the named array and nothing else.
	EXPECT_TRUE(gvl.add_primitives(ePRIM_SPHERE, "y"));
	EXPECT_TRUE(gvl.modify_primitives("y", a));
	EXPECT_EQ(gvl.get_primitives("y")->size(), 1u);
	EXPECT_EQ(gvl.get_primitives("x")->size(), 0u);
	EXPECT_TRUE(gvl.modify_primitives("y", std::vector<Coord4i>{ { 1, 1, 1, 1 }, { 2, 2, 2, 2 } }));
	EXPECT_EQ(gvl.get_primitives("y")->size(), 2u);
	EXPECT_TRUE(gvl.modify_primitives("y", std::vector<Point3f>{ { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } }, 1.0f));
	EXPECT_EQ(gvl.get_primitives("y")->size(), 3u);
	EXPECT_TRUE(gvl.modify_primitives("y", std::vector<Coord3i>{ { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 }, { 4, 4, 4 } }, 1));
	EXPECT_EQ(gvl.get_primitives("y")->size(), 4u);

	// A refused modification leaves the array as it was; a handle outlives del_primitives.
	const PrimitiveArraySharedPtr handle = gvl.get_primitives("y");
	const auto before = entries(*handle);
	expect_error([&] { (void)gvl.modify_primitives("y", std::vector<Coord4i>{ { 1, 1, 1, -1 } }); }, "the size of primitive 0 is negative");
	EXPECT_EQ(entries(*handle), before);
	EXPECT_TRUE(gvl.del_primitives("y"));
	EXPECT_FALSE(gvl.del_primitives("y"));
	EXPECT_EQ(gvl.get_primitives("y"), nullptr);
	EXPECT_EQ(entries(*handle), before) << "the array lives as long as a handle exists";
	EXPECT_TRUE(gvl.add_primitives(ePRIM_CUBOID, "y")) << "the name is free again";
	EXPECT_EQ(gvl.get_primitives("y")->size(), 0u);
}
