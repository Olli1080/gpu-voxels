// In-process viewer core (Viewer.h): GPU extraction of cubes from the maps and lists, and the renderer. Rendering is checked on an offscreen image:
// the pixels at the projected position of a cube (computed on the host from the camera matrix) have the cube's colour, everything outside its
// silhouette has the clear colour, nearer cubes hide farther ones, and shaded output differs between the faces of a cube.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/ImageFile.h>
#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/OrbitCamera.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/Viewer.h>
#include <gpu_voxels/vulkan/VoxelList.h>

#include "../reference/octree_reference.h"

using namespace gpu_voxels::vulkan;

namespace
{
	/// One Context with a graphics + compute queue for all viewer tests (the shared context of the other tests is compute only).
	struct ViewerContext
	{
		std::unique_ptr<Context> context;
		std::string error;
		bool no_device = false;
	};

	ViewerContext& viewer_context()
	{
		static ViewerContext result = []
		{
			ViewerContext r;
			try
			{
				ContextCreateInfo info;
				info.enable_graphics = true;
				info.require_validation = std::getenv("GVL_ALLOW_NO_VALIDATION") == nullptr;
				r.context = std::make_unique<Context>(info);
			}
			catch (const NoDeviceError& e)
			{
				r.no_device = true;
				r.error = e.what();
			}
			catch (const std::exception& e)
			{
				r.error = e.what();
			}
			return r;
		}();
		return result;
	}

	class ViewerTest : public ::testing::Test
	{
	protected:
		void SetUp() override
		{
			auto& v = viewer_context();
			if (v.no_device)
				GTEST_SKIP() << "No usable Vulkan device with a graphics queue: " << v.error;
			ASSERT_TRUE(v.context != nullptr) << "Vulkan context creation failed: " << v.error;
			ctx().reset_validation_state();
		}

		void TearDown() override
		{
			if (!viewer_context().context)
				return;
			std::string messages;
			for (const auto& m : ctx().validation_messages())
				messages += m + "\n";
			EXPECT_EQ(ctx().validation_error_count(), 0u) << "Vulkan validation errors:\n" << messages;
		}

		[[nodiscard]] Context& ctx() { return *viewer_context().context; }
	};

	using Cube = CubeBuffer::Cube;

	std::vector<Cube> sorted(std::vector<Cube> cubes)
	{
		std::sort(cubes.begin(), cubes.end(), [](const Cube& a, const Cube& b) { return std::tie(a.z, a.y, a.x) < std::tie(b.z, b.y, b.x); });
		return cubes;
	}

	struct Pixel
	{
		int r, g, b, a;
		bool operator==(const Pixel&) const = default;
	};

	Pixel pixel(const std::vector<uint8_t>& rgba, uint32_t width, int x, int y)
	{
		const std::size_t i = (static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)) * 4;
		return { rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3] };
	}

	Pixel pixel_of(uint32_t packed) { return { int(packed & 255u), int((packed >> 8) & 255u), int((packed >> 16) & 255u), 255 }; }

	bool near(const Pixel& a, const Pixel& b, int tolerance)
	{
		return std::abs(a.r - b.r) <= tolerance && std::abs(a.g - b.g) <= tolerance && std::abs(a.b - b.b) <= tolerance;
	}

	/// Pixel coordinates (x right, y down) of a world point for the camera and the image size.
	std::array<float, 3> project(const ViewerCamera& camera, uint32_t width, uint32_t height, std::array<float, 3> p)
	{
		const auto m = camera.view_projection(static_cast<float>(width) / static_cast<float>(height));
		const float x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
		const float y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
		const float z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
		const float w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
		return { (x / w * 0.5f + 0.5f) * static_cast<float>(width), (y / w * 0.5f + 0.5f) * static_cast<float>(height), z / w };
	}

	RenderOptions flat_black()
	{
		RenderOptions options;
		options.clear_colour = { 0.0f, 0.0f, 0.0f, 1.0f };
		options.shaded = false;
		return options;
	}

	std::vector<Cube> expected_bit_cubes(const BitVoxelMap& map, const BitPalette& palette)
	{
		const auto words = map.download_words();
		const Dim3ui d = map.dimensions();
		std::vector<Cube> cubes;
		for (uint64_t i = 0; i < d.voxel_count(); ++i)
		{
			uint32_t colour = 0;
			for (uint32_t meaning = 0; meaning < 256 && colour == 0; ++meaning)
				if (((words[i * 8 + meaning / 32] >> (meaning % 32)) & 1u) != 0 && (palette.colour[meaning] >> 24) != 0)
					colour = palette.colour[meaning];
			if (colour == 0)
				continue;
			const float side = map.voxel_side_length();
			cubes.push_back({ static_cast<float>(i % d.x) * side, static_cast<float>((i / d.x) % d.y) * side, static_cast<float>(i / (static_cast<uint64_t>(d.x) * d.y)) * side, side, colour });
		}
		return cubes;
	}

	void expect_equal(const std::vector<Cube>& actual, const std::vector<Cube>& expected)
	{
		const auto a = sorted(actual);
		const auto e = sorted(expected);
		ASSERT_EQ(a.size(), e.size());
		for (std::size_t i = 0; i < a.size(); ++i)
		{
			EXPECT_EQ(a[i].x, e[i].x) << i;
			EXPECT_EQ(a[i].y, e[i].y) << i;
			EXPECT_EQ(a[i].z, e[i].z) << i;
			EXPECT_EQ(a[i].size, e[i].size) << i;
			EXPECT_EQ(a[i].colour, e[i].colour) << i;
		}
	}
}

// ---- camera ----------------------------------------------------------------------------------------------------------------------------------

TEST(ViewerCameraTest, ProjectsLikeAVulkanCamera)
{
	ViewerCamera camera;
	camera.eye = { 0.0f, 0.0f, 5.0f };
	camera.target = { 0.0f, 0.0f, 0.0f };
	const auto centre = project(camera, 100, 100, { 0.0f, 0.0f, 0.0f });
	EXPECT_NEAR(centre[0], 50.0f, 1e-3f);
	EXPECT_NEAR(centre[1], 50.0f, 1e-3f);
	EXPECT_GT(centre[2], 0.0f);
	EXPECT_LT(centre[2], 1.0f);
	EXPECT_GT(project(camera, 100, 100, { 1.0f, 0.0f, 0.0f })[0], 50.0f) << "+x is to the right";
	EXPECT_LT(project(camera, 100, 100, { 0.0f, 1.0f, 0.0f })[1], 50.0f) << "+y is up: smaller pixel row";
	// a point at distance d with half the field of view above the axis lands on the top edge
	const float half = 5.0f * std::tan(30.0f * 3.14159265f / 180.0f);
	EXPECT_NEAR(project(camera, 100, 100, { 0.0f, half, 0.0f })[1], 0.0f, 0.05f);
	// near and far plane map to depth 0 and 1
	EXPECT_NEAR(project(camera, 100, 100, { 0.0f, 0.0f, 5.0f - camera.z_near })[2], 0.0f, 1e-4f);
	EXPECT_NEAR(project(camera, 100, 100, { 0.0f, 0.0f, 5.0f - camera.z_far })[2], 1.0f, 1e-4f);
	// the aspect ratio widens the horizontal field of view
	EXPECT_NEAR(project(camera, 200, 100, { half * 2.0f, 0.0f, 0.0f })[0], 200.0f, 0.1f);
	camera.target = camera.eye;
	EXPECT_THROW((void)camera.view_projection(1.0f), VulkanError);
}

TEST(ViewerPaletteTest, DefaultsHideFreeAndAreDistinct)
{
	const BitPalette palette = BitPalette::defaults();
	EXPECT_EQ(palette.colour[0], 0u) << "eBVM_FREE is not drawn";
	std::set<uint32_t> distinct;
	for (uint32_t m = 1; m < 256; ++m)
	{
		EXPECT_EQ(palette.colour[m] >> 24, 255u) << m;
		distinct.insert(palette.colour[m]);
	}
	EXPECT_GE(distinct.size(), 250u) << "the golden-angle hues give (nearly) distinct colours";
}

// ---- extraction ------------------------------------------------------------------------------------------------------------------------------

TEST_F(ViewerTest, ExtractsTheVisibleMeaningsOfABitMap)
{
	BitVoxelMap map(ctx(), { 8, 6, 4 }, 0.5f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 2, 3 }, { 7, 5, 0 } }, 5);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 2, 3 }, { 0, 0, 0 } }, 40);  // second word of the voxel; (1,2,3) has 5 and 40
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 4, 4, 1 } }, 0);                // FREE only: hidden
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 4, 4, 1 }, { 2, 0, 2 } }, 255); // the last meaning
	const BitPalette palette = BitPalette::defaults();

	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 100);
	extractor.extract(map, palette, cubes);
	EXPECT_EQ(cubes.count(), 5u);
	EXPECT_EQ(cubes.found(), 5u);
	EXPECT_EQ(cubes.dropped(), 0u);
	// voxel (7,5,0) -> 3.5, 2.5, 0; (0,0,0) -> 0; (2,0,2) -> 1, 0, 1; (4,4,1) -> 2, 2, 0.5; (1,2,3) -> 0.5, 1, 1.5 (side 0.5)
	expect_equal(cubes.download(), {
		{ 0.0f, 0.0f, 0.0f, 0.5f, palette.colour[40] },
		{ 1.0f, 0.0f, 1.0f, 0.5f, palette.colour[255] },
		{ 2.0f, 2.0f, 0.5f, 0.5f, palette.colour[255] }, // FREE (0) is hidden, so the lowest visible meaning is 255
		{ 3.5f, 2.5f, 0.0f, 0.5f, palette.colour[5] },
		{ 0.5f, 1.0f, 1.5f, 0.5f, palette.colour[5] } });
}

TEST_F(ViewerTest, BitMapExtractionMatchesAHostModelOnRandomContent)
{
	BitVoxelMap map(ctx(), { 40, 30, 20 }, 0.25f);
	std::srand(7);
	for (const uint8_t meaning : { uint8_t(1), uint8_t(3), uint8_t(33), uint8_t(64), uint8_t(129), uint8_t(200), uint8_t(0) })
	{
		std::vector<Coord3ui> coords;
		for (int i = 0; i < 1500; ++i)
			coords.push_back({ uint32_t(std::rand() % 40), uint32_t(std::rand() % 30), uint32_t(std::rand() % 20) });
		map.insert_coordinate_list(coords, meaning);
	}
	BitPalette palette = BitPalette::defaults();
	palette.hide(3);
	palette.hide(64);
	palette.set(200, pack_colour(1, 2, 3, 4));
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 24000);
	extractor.extract(map, palette, cubes);
	const auto expected = expected_bit_cubes(map, palette);
	EXPECT_GT(expected.size(), 2000u);
	EXPECT_EQ(cubes.found(), expected.size());
	expect_equal(cubes.download(), expected);
}

TEST_F(ViewerTest, CubesBeyondTheCapacityAreCountedNotStored)
{
	BitVoxelMap map(ctx(), { 4, 4, 4 }, 1.0f);
	std::vector<Coord3ui> all;
	for (uint32_t z = 0; z < 4; ++z)
		for (uint32_t y = 0; y < 4; ++y)
			for (uint32_t x = 0; x < 4; ++x)
				all.push_back({ x, y, z });
	map.insert_coordinate_list(all, 5);
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 10);
	extractor.extract(map, BitPalette::defaults(), cubes);
	EXPECT_EQ(cubes.found(), 64u);
	EXPECT_EQ(cubes.count(), 10u);
	EXPECT_EQ(cubes.dropped(), 54u);
	const auto stored = cubes.download();
	EXPECT_EQ(stored.size(), 10u);
	for (const Cube& c : stored)
		EXPECT_EQ(c.colour, BitPalette::defaults().colour[5]);
	// drawing a full buffer must stay inside the instance buffer (validation checks the indirect draw)
	VoxelRenderer renderer(ctx(), OffscreenTarget::kColourFormat);
	OffscreenTarget target(ctx(), 32, 32);
	ViewerCamera camera;
	camera.eye = { 2.0f, 2.0f, 10.0f };
	camera.target = { 2.0f, 2.0f, 2.0f };
	const CubeBuffer* buffers[] = { &cubes };
	target.render(renderer, camera, buffers, flat_black());
	// a second extraction replaces the content: an empty map gives nothing
	BitVoxelMap empty(ctx(), { 4, 4, 4 }, 1.0f);
	extractor.extract(empty, BitPalette::defaults(), cubes);
	EXPECT_EQ(cubes.count(), 0u);
	EXPECT_EQ(cubes.found(), 0u);
}

TEST_F(ViewerTest, ExtractsTheOccupiedVoxelsOfAProbMap)
{
	ProbVoxelMap map(ctx(), { 10, 10, 10 }, 2.0f);
	const auto at = [](float x, float y, float z) { return std::vector<Point3f>{ { x * 2.0f + 1.0f, y * 2.0f + 1.0f, z * 2.0f + 1.0f } }; };
	map.insert_point_cloud(at(1, 1, 1), eBVM_OCCUPIED);     // +72
	map.insert_point_cloud(at(2, 1, 1), eBVM_OCCUPIED);
	map.insert_point_cloud(at(2, 1, 1), eBVM_OCCUPIED);     // clamps at 127
	map.insert_point_cloud(at(3, 1, 1), eBVM_FREE);         // -10: below the default threshold
	map.insert_point_cloud(at(4, 1, 1), 131);               // swept volume meaning 131: +2
	map.insert_point_cloud(at(5, 1, 1), eBVM_UNKNOWN);      // unknown stays unknown
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 1000);
	extractor.extract(map, ProbDisplay{}, cubes);

	const auto occupancy = map.download_occupancy();
	std::vector<std::pair<std::array<float, 3>, int>> expected;
	for (std::size_t i = 0; i < occupancy.size(); ++i)
		if (occupancy[i] != -128 && occupancy[i] >= 1)
			expected.push_back({ { float(i % 10) * 2.0f, float((i / 10) % 10) * 2.0f, float(i / 100) * 2.0f }, occupancy[i] });
	ASSERT_EQ(expected.size(), 3u) << "the voxels with +72, +127 and +2";
	const auto actual = sorted(cubes.download());
	ASSERT_EQ(actual.size(), 3u);
	for (std::size_t i = 0; i < 3; ++i)
	{
		EXPECT_EQ(actual[i].size, 2.0f);
		EXPECT_EQ(actual[i].x, expected[i].first[0]) << i;
		EXPECT_EQ(actual[i].y, expected[i].first[1]) << i;
		EXPECT_EQ(actual[i].z, expected[i].first[2]) << i;
		// colour: the gradient from blue (-127) to red (+127)
		const float t = static_cast<float>(expected[i].second + 127) / 254.0f;
		const auto channel = [](float v) { return std::clamp(v, 0.0f, 1.0f); };
		const Pixel want{ int(std::lround(channel(1.5f - std::fabs(4.0f * t - 3.0f)) * 255.0f)), int(std::lround(channel(1.5f - std::fabs(4.0f * t - 2.0f)) * 255.0f)),
			int(std::lround(channel(1.5f - std::fabs(4.0f * t - 1.0f)) * 255.0f)), 255 };
		EXPECT_TRUE(near(pixel_of(actual[i].colour), want, 1)) << i << ": value " << expected[i].second;
	}
	// a lower threshold also shows the free voxel
	extractor.extract(map, ProbDisplay{ -20 }, cubes);
	EXPECT_EQ(cubes.count(), 4u);
	// even a threshold below every value never shows the unknown voxels (-128): only the four voxels that were touched
	extractor.extract(map, ProbDisplay{ -1000 }, cubes);
	EXPECT_EQ(cubes.count(), 4u);
	// the highest threshold shows only the saturated voxel
	extractor.extract(map, ProbDisplay{ 127 }, cubes);
	EXPECT_EQ(cubes.count(), 1u);
}

TEST_F(ViewerTest, ExtractsTheEntriesOfABitList)
{
	BitVoxelList list(ctx(), { 16, 16, 16 }, 0.25f);
	list.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 2, 3 }, { 15, 15, 15 }, { 0, 0, 0 } }, 7);
	list.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 2, 3 }, { 9, 8, 7 } }, 70);
	list.insert_coordinate_list(std::vector<Coord3ui>{ { 4, 4, 4 } }, 0); // FREE only: hidden
	ASSERT_EQ(list.size(), 5u);
	const BitPalette palette = BitPalette::defaults();
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 100);
	extractor.extract(list, palette, cubes);
	expect_equal(cubes.download(), {
		{ 0.25f, 0.5f, 0.75f, 0.25f, palette.colour[7] },
		{ 3.75f, 3.75f, 3.75f, 0.25f, palette.colour[7] },
		{ 0.0f, 0.0f, 0.0f, 0.25f, palette.colour[7] },
		{ 2.25f, 2.0f, 1.75f, 0.25f, palette.colour[70] } });
	// an empty list gives an empty buffer
	BitVoxelList empty(ctx(), { 16, 16, 16 }, 0.25f);
	extractor.extract(empty, palette, cubes);
	EXPECT_EQ(cubes.count(), 0u);
}

// ---- rendering -------------------------------------------------------------------------------------------------------------------------------

TEST_F(ViewerTest, ADrawnCubeCoversExactlyItsProjection)
{
	BitVoxelMap map(ctx(), { 8, 8, 8 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 3, 2, 1 } }, 5);
	const BitPalette palette = BitPalette::defaults();
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 10);
	extractor.extract(map, palette, cubes);

	VoxelRenderer renderer(ctx(), OffscreenTarget::kColourFormat);
	constexpr uint32_t w = 128, h = 96;
	OffscreenTarget target(ctx(), w, h);
	ViewerCamera camera;
	camera.eye = { 3.5f, 2.5f, 5.0f };
	camera.target = { 3.5f, 2.5f, 1.5f };
	const CubeBuffer* buffers[] = { &cubes };
	target.render(renderer, camera, buffers, flat_black());
	const auto image = target.read_rgba();
	ASSERT_EQ(image.size(), size_t(w) * h * 4);

	// the front face z = 2 is the silhouette: its corners in pixels
	const auto lo = project(camera, w, h, { 3.0f, 3.0f, 2.0f });
	const auto hi = project(camera, w, h, { 4.0f, 2.0f, 2.0f });
	ASSERT_LT(lo[0], hi[0]);
	ASSERT_LT(lo[1], hi[1]);
	const Pixel colour = pixel_of(palette.colour[5]);
	const Pixel clear{ 0, 0, 0, 255 };
	for (const float fx : { 0.5f, 0.1f, 0.9f })
		for (const float fy : { 0.5f, 0.1f, 0.9f })
		{
			const int px = static_cast<int>(lo[0] + (hi[0] - lo[0]) * fx), py = static_cast<int>(lo[1] + (hi[1] - lo[1]) * fy);
			EXPECT_EQ(pixel(image, w, px, py), colour) << px << "," << py;
		}
	for (const auto& [px, py] : { std::pair{ int(lo[0]) - 3, int(lo[1]) + 5 }, std::pair{ int(hi[0]) + 3, int(hi[1]) - 5 }, std::pair{ int(lo[0]) + 5, int(lo[1]) - 3 },
		     std::pair{ int(hi[0]) - 5, int(hi[1]) + 3 }, std::pair{ 0, 0 }, std::pair{ int(w) - 1, int(h) - 1 } })
		EXPECT_EQ(pixel(image, w, px, py), clear) << px << "," << py;
	// exactly the pixels of the front face are covered (the rasterised square is as large as its projection, give or take the border pixels)
	std::size_t covered = 0;
	for (uint32_t y = 0; y < h; ++y)
		for (uint32_t x = 0; x < w; ++x)
			covered += pixel(image, w, int(x), int(y)) == colour ? 1 : 0;
	const double expected_area = double(hi[0] - lo[0]) * double(hi[1] - lo[1]);
	EXPECT_NEAR(double(covered), expected_area, 2.0 * (double(hi[0] - lo[0]) + double(hi[1] - lo[1])));
}

TEST_F(ViewerTest, TheNearerCubeHidesTheFartherOne)
{
	BitVoxelMap map(ctx(), { 4, 4, 4 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 0, 0, 0 } }, 5);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 0, 0, 3 } }, 6);
	const BitPalette palette = BitPalette::defaults();
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 10);
	extractor.extract(map, palette, cubes);
	VoxelRenderer renderer(ctx(), OffscreenTarget::kColourFormat);
	OffscreenTarget target(ctx(), 64, 64);
	const CubeBuffer* buffers[] = { &cubes };

	ViewerCamera camera;
	camera.eye = { 0.5f, 0.5f, 12.0f };
	camera.target = { 0.5f, 0.5f, 0.0f };
	target.render(renderer, camera, buffers, flat_black());
	EXPECT_EQ(pixel(target.read_rgba(), 64, 32, 32), pixel_of(palette.colour[6])) << "seen from +z the cube at z = 3 is in front";
	camera.eye = { 0.5f, 0.5f, -12.0f };
	target.render(renderer, camera, buffers, flat_black());
	EXPECT_EQ(pixel(target.read_rgba(), 64, 32, 32), pixel_of(palette.colour[5])) << "seen from -z the cube at z = 0 is in front";
}

TEST_F(ViewerTest, ShadingGivesTheThreeVisibleFacesDifferentBrightness)
{
	BitVoxelMap map(ctx(), { 4, 4, 4 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 1, 1 } }, 5);
	const BitPalette palette = BitPalette::defaults();
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 10);
	extractor.extract(map, palette, cubes);
	VoxelRenderer renderer(ctx(), OffscreenTarget::kColourFormat);
	OffscreenTarget target(ctx(), 96, 96);
	ViewerCamera camera;
	camera.eye = { 6.0f, 5.0f, 7.0f };
	camera.target = { 1.5f, 1.5f, 1.5f };
	const CubeBuffer* buffers[] = { &cubes };

	std::set<std::tuple<int, int, int>> shades;
	RenderOptions shaded = flat_black();
	shaded.shaded = true;
	target.render(renderer, camera, buffers, shaded);
	auto image = target.read_rgba();
	const Pixel flat = pixel_of(palette.colour[5]);
	int brightest = 0;
	for (std::size_t i = 0; i < image.size(); i += 4)
		if (image[i] || image[i + 1] || image[i + 2])
		{
			shades.insert({ image[i], image[i + 1], image[i + 2] });
			brightest = std::max<int>(brightest, image[i + 1]);
			EXPECT_LE(image[i], flat.r);
			EXPECT_LE(image[i + 1], flat.g);
			EXPECT_LE(image[i + 2], flat.b);
		}
	EXPECT_EQ(shades.size(), 3u) << "+x, +y and +z faces";
	// the +y face faces the light most: 0.35 + 0.65 * 0.799
	EXPECT_NEAR(brightest, flat.g * 0.8695, 2.0);

	// unshaded: exactly the palette colour everywhere on the cube
	target.render(renderer, camera, buffers, flat_black());
	image = target.read_rgba();
	std::set<std::tuple<int, int, int>> flat_shades;
	for (std::size_t i = 0; i < image.size(); i += 4)
		if (image[i] || image[i + 1] || image[i + 2])
			flat_shades.insert({ image[i], image[i + 1], image[i + 2] });
	EXPECT_EQ(flat_shades, (std::set<std::tuple<int, int, int>>{ { flat.r, flat.g, flat.b } }));
}

TEST_F(ViewerTest, SeveralBuffersAreDrawnIntoOneImage)
{
	BitVoxelMap map(ctx(), { 8, 8, 8 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 1, 1, 1 } }, 5);
	BitVoxelList list(ctx(), { 8, 8, 8 }, 1.0f);
	list.insert_coordinate_list(std::vector<Coord3ui>{ { 5, 1, 1 } }, 9);
	ProbVoxelMap prob(ctx(), { 8, 8, 8 }, 1.0f);
	prob.insert_point_cloud(std::vector<Point3f>{ { 3.5f, 5.5f, 1.5f } }, eBVM_OCCUPIED);
	const BitPalette palette = BitPalette::defaults();
	CubeExtractor extractor(ctx());
	CubeBuffer a(ctx(), 10), b(ctx(), 10), c(ctx(), 10);
	extractor.extract(map, palette, a);
	extractor.extract(list, palette, b);
	extractor.extract(prob, ProbDisplay{}, c);

	VoxelRenderer renderer(ctx(), OffscreenTarget::kColourFormat);
	constexpr uint32_t w = 128, h = 128;
	OffscreenTarget target(ctx(), w, h);
	ViewerCamera camera;
	camera.eye = { 3.5f, 3.5f, 14.0f };
	camera.target = { 3.5f, 3.5f, 1.5f };
	const CubeBuffer* buffers[] = { &a, &b, &c };
	target.render(renderer, camera, buffers, flat_black());
	const auto image = target.read_rgba();
	const auto centre_of = [&](float x, float y)
	{
		const auto p = project(camera, w, h, { x, y, 2.0f });
		return pixel(image, w, int(p[0]), int(p[1]));
	};
	EXPECT_EQ(centre_of(1.5f, 1.5f), pixel_of(palette.colour[5]));
	EXPECT_EQ(centre_of(5.5f, 1.5f), pixel_of(palette.colour[9]));
	EXPECT_FALSE(near(centre_of(3.5f, 5.5f), Pixel{ 0, 0, 0, 255 }, 3)) << "the probabilistic voxel";
	EXPECT_EQ(centre_of(6.5f, 6.5f), (Pixel{ 0, 0, 0, 255 })) << "empty space";
}

TEST_F(ViewerTest, TheRendererNeedsAGraphicsContext)
{
	// the shared compute-only context of the other tests (a different Context than the one of this fixture)
	ContextCreateInfo info;
	info.require_validation = std::getenv("GVL_ALLOW_NO_VALIDATION") == nullptr;
	Context compute_only(info);
	EXPECT_FALSE(compute_only.info().graphics);
	EXPECT_THROW(VoxelRenderer(compute_only, OffscreenTarget::kColourFormat), VulkanError);
	EXPECT_TRUE(ctx().info().graphics);
	// a renderer for another format cannot draw into the offscreen target
	VoxelRenderer other_format(ctx(), vk::Format::eB8G8R8A8Unorm);
	OffscreenTarget target(ctx(), 16, 16);
	EXPECT_THROW(target.render(other_format, ViewerCamera{}, {}, RenderOptions{}), VulkanError);
}

// ---- orbit camera and controller (pure maths) ------------------------------------------------------------------------------------------------

namespace
{
	float length(const std::array<float, 3>& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
	std::array<float, 3> minus(const std::array<float, 3>& a, const std::array<float, 3>& b) { return { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; }
	float dot3(const std::array<float, 3>& a, const std::array<float, 3>& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
}

TEST(OrbitCameraTest, TheEyeIsOnASphereAroundTheTargetAndZIsUp)
{
	OrbitCamera camera;
	camera.target = { 1.0f, 2.0f, 3.0f };
	camera.distance = 5.0f;
	camera.yaw_degrees = 30.0f;
	camera.pitch_degrees = 20.0f;
	const auto eye = camera.eye();
	EXPECT_NEAR(length(minus(eye, camera.target)), 5.0f, 1e-4f);
	EXPECT_NEAR(eye[2], 3.0f + 5.0f * std::sin(20.0f * 3.14159265f / 180.0f), 1e-4f);
	const ViewerCamera view = camera.camera();
	EXPECT_EQ(view.up, (std::array<float, 3>{ 0.0f, 0.0f, 1.0f }));
	EXPECT_EQ(view.target, camera.target);
	EXPECT_EQ(view.eye, eye);
	EXPECT_LT(view.z_near, view.z_far);
	// yaw 0, pitch 0: the eye is on the +x side
	camera.yaw_degrees = 0.0f;
	camera.pitch_degrees = 0.0f;
	EXPECT_NEAR(camera.eye()[0], 6.0f, 1e-4f);
	EXPECT_NEAR(camera.eye()[1], 2.0f, 1e-4f);
}

TEST(OrbitCameraTest, RotateZoomAndPan)
{
	OrbitCamera camera;
	camera.yaw_degrees = 10.0f;
	camera.pitch_degrees = 20.0f;
	camera.distance = 8.0f;
	camera.rotate(100.0f, 50.0f);
	EXPECT_NEAR(camera.yaw_degrees, 10.0f - 30.0f, 1e-4f) << "dragging right turns the scene to the right: the eye goes left";
	EXPECT_NEAR(camera.pitch_degrees, 20.0f + 15.0f, 1e-4f);
	camera.rotate(0.0f, 100000.0f);
	EXPECT_EQ(camera.pitch_degrees, 89.0f) << "the pitch stops short of the poles";
	camera.rotate(0.0f, -100000.0f);
	EXPECT_EQ(camera.pitch_degrees, -89.0f);
	EXPECT_NEAR(length(minus(camera.eye(), camera.target)), 8.0f, 1e-4f);

	camera.zoom(1.0f);
	EXPECT_NEAR(camera.distance, 8.0f / 1.15f, 1e-4f);
	camera.zoom(-1.0f);
	EXPECT_NEAR(camera.distance, 8.0f, 1e-4f);
	camera.zoom(1000.0f);
	EXPECT_GT(camera.distance, 0.0f) << "never reaches the target";

	// panning: perpendicular to the view direction, to the left for a drag to the right, one pixel is distance * 2 tan(fov / 2) / height
	OrbitCamera pan;
	pan.distance = 10.0f;
	pan.yaw_degrees = 35.0f;
	pan.pitch_degrees = 25.0f;
	const auto forward = minus(pan.target, pan.eye());
	const auto before = pan.target;
	pan.pan(30.0f, 0.0f, 600.0f);
	const auto delta = minus(pan.target, before);
	EXPECT_NEAR(dot3(delta, forward), 0.0f, 1e-4f);
	const float per_pixel = 2.0f * 10.0f * std::tan(30.0f * 3.14159265f / 180.0f) / 600.0f;
	EXPECT_NEAR(length(delta), 30.0f * per_pixel, 1e-4f);
	const std::array<float, 3> right{ -std::sin(35.0f * 3.14159265f / 180.0f), std::cos(35.0f * 3.14159265f / 180.0f), 0.0f };
	EXPECT_LT(dot3(delta, right), 0.0f) << "the scene follows the cursor: the target moves against it";
	const auto moved = pan.target;
	pan.pan(5.0f, 5.0f, 0.0f); // an empty viewport changes nothing
	EXPECT_EQ(pan.target, moved);
}

TEST(OrbitCameraTest, FrameFitsTheBoxIntoTheView)
{
	OrbitCamera camera;
	camera.yaw_degrees = 40.0f;
	camera.pitch_degrees = 30.0f;
	const std::array<float, 3> lo{ 0.0f, 0.0f, 0.0f }, hi{ 4.0f, 2.0f, 1.0f };
	camera.frame(lo, hi);
	EXPECT_EQ(camera.target, (std::array<float, 3>{ 2.0f, 1.0f, 0.5f }));
	const ViewerCamera view = camera.camera();
	float biggest = 0.0f;
	const auto m = view.view_projection(1.0f);
	for (int i = 0; i < 8; ++i)
	{
		const float x = (i & 1) ? hi[0] : lo[0], y = (i & 2) ? hi[1] : lo[1], z = (i & 4) ? hi[2] : lo[2];
		const float cx = m[0] * x + m[4] * y + m[8] * z + m[12], cy = m[1] * x + m[5] * y + m[9] * z + m[13];
		const float w = m[3] * x + m[7] * y + m[11] * z + m[15];
		EXPECT_GT(w, 0.0f);
		EXPECT_LE(std::fabs(cx / w), 1.0f) << i;
		EXPECT_LE(std::fabs(cy / w), 1.0f) << i;
		biggest = std::max({ biggest, std::fabs(cx / w), std::fabs(cy / w) });
	}
	EXPECT_GT(biggest, 0.4f) << "the box fills a good part of the view";
	// a degenerate box still gives a usable camera
	camera.frame({ 1, 1, 1 }, { 1, 1, 1 });
	EXPECT_GT(camera.distance, 0.0f);
}

TEST(OrbitControllerTest, ButtonsSelectRotatePanOrNothing)
{
	OrbitCamera camera;
	OrbitController controller(camera);
	const OrbitCamera start = camera;
	controller.cursor(100.0f, 100.0f, 600.0f);
	controller.cursor(150.0f, 120.0f, 600.0f); // no button: only the position is tracked
	EXPECT_EQ(camera.yaw_degrees, start.yaw_degrees);
	EXPECT_EQ(camera.target, start.target);

	controller.button(MouseButton::Left, true);
	controller.cursor(160.0f, 120.0f, 600.0f); // 10 px right
	EXPECT_NEAR(camera.yaw_degrees, start.yaw_degrees - 3.0f, 1e-4f);
	EXPECT_EQ(camera.target, start.target);
	controller.button(MouseButton::Left, false);

	const float yaw = camera.yaw_degrees;
	controller.button(MouseButton::Right, true);
	controller.cursor(180.0f, 120.0f, 600.0f);
	EXPECT_EQ(camera.yaw_degrees, yaw);
	EXPECT_NE(camera.target, start.target);
	controller.button(MouseButton::Right, false);

	const float distance = camera.distance;
	controller.scroll(2.0f);
	EXPECT_LT(camera.distance, distance);
}

// ---- screenshots -----------------------------------------------------------------------------------------------------------------------------

TEST(ImageFileTest, EncodesAValidPng)
{
	constexpr uint32_t w = 5, h = 3;
	std::vector<uint8_t> rgba(w * h * 4);
	for (std::size_t i = 0; i < rgba.size(); ++i)
		rgba[i] = static_cast<uint8_t>(i * 7 + 3);
	const auto png = encode_png(w, h, rgba);

	ASSERT_GT(png.size(), 8u + 25u + 12u);
	EXPECT_EQ(std::vector<uint8_t>(png.begin(), png.begin() + 8), (std::vector<uint8_t>{ 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A }));

	// walk the chunks: lengths, CRC32 over type + data, the header, and the stored deflate blocks of the data
	const auto be32 = [&](std::size_t at) { return (uint32_t(png[at]) << 24) | (uint32_t(png[at + 1]) << 16) | (uint32_t(png[at + 2]) << 8) | uint32_t(png[at + 3]); };
	const auto crc = [](const uint8_t* data, std::size_t n)
	{
		uint32_t c = 0xFFFFFFFFu;
		for (std::size_t i = 0; i < n; ++i)
		{
			c ^= data[i];
			for (int k = 0; k < 8; ++k)
				c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
		}
		return ~c;
	};
	std::vector<std::string> types;
	std::vector<uint8_t> idat;
	for (std::size_t pos = 8; pos < png.size();)
	{
		const uint32_t n = be32(pos);
		ASSERT_LE(pos + 12 + n, png.size());
		types.emplace_back(reinterpret_cast<const char*>(&png[pos + 4]), 4);
		EXPECT_EQ(be32(pos + 8 + n), crc(&png[pos + 4], 4 + n)) << types.back();
		if (types.back() == "IHDR")
		{
			EXPECT_EQ(n, 13u);
			EXPECT_EQ(be32(pos + 8), w);
			EXPECT_EQ(be32(pos + 12), h);
			EXPECT_EQ(png[pos + 16], 8);
			EXPECT_EQ(png[pos + 17], 6);
		}
		if (types.back() == "IDAT")
			idat.insert(idat.end(), png.begin() + static_cast<std::ptrdiff_t>(pos + 8), png.begin() + static_cast<std::ptrdiff_t>(pos + 8 + n));
		pos += 12 + n;
	}
	EXPECT_EQ(types, (std::vector<std::string>{ "IHDR", "IDAT", "IEND" }));

	// zlib header, stored blocks, Adler-32
	ASSERT_GT(idat.size(), 2u + 5u + 4u);
	EXPECT_EQ(((idat[0] << 8) | idat[1]) % 31, 0);
	std::vector<uint8_t> raw;
	std::size_t at = 2;
	for (bool last = false; !last;)
	{
		last = (idat[at] & 1) != 0;
		EXPECT_EQ(idat[at] & 6, 0) << "stored block";
		const std::size_t n = idat[at + 1] | (idat[at + 2] << 8);
		EXPECT_EQ(uint16_t(idat[at + 3] | (idat[at + 4] << 8)), uint16_t(~n));
		raw.insert(raw.end(), idat.begin() + static_cast<std::ptrdiff_t>(at + 5), idat.begin() + static_cast<std::ptrdiff_t>(at + 5 + n));
		at += 5 + n;
	}
	ASSERT_EQ(raw.size(), (w * 4 + 1) * h);
	uint32_t a = 1, b = 0;
	for (const uint8_t byte : raw)
	{
		a = (a + byte) % 65521u;
		b = (b + a) % 65521u;
	}
	EXPECT_EQ((uint32_t(idat[at]) << 24) | (uint32_t(idat[at + 1]) << 16) | (uint32_t(idat[at + 2]) << 8) | idat[at + 3], (b << 16) | a);
	for (uint32_t y = 0; y < h; ++y)
	{
		EXPECT_EQ(raw[y * (w * 4 + 1)], 0) << "filter type none";
		for (uint32_t x = 0; x < w * 4; ++x)
			EXPECT_EQ(raw[y * (w * 4 + 1) + 1 + x], rgba[y * w * 4 + x]);
	}

	EXPECT_THROW((void)encode_png(w, h, std::span<const uint8_t>(rgba).first(10)), VulkanError);
	EXPECT_THROW((void)encode_png(0, h, rgba), VulkanError);
}

TEST(ImageFileTest, LargeImagesUseSeveralStoredBlocks)
{
	// 300 x 100 x 4 bytes = 120 KB: more than one block of 65535 bytes
	std::vector<uint8_t> rgba(300 * 100 * 4, 0x55);
	const auto png = encode_png(300, 100, rgba);
	EXPECT_GT(png.size(), rgba.size());
	EXPECT_LT(png.size(), rgba.size() + 200);
}

TEST_F(ViewerTest, AnOrbitCameraLooksAtItsTarget)
{
	BitVoxelMap map(ctx(), { 8, 8, 8 }, 1.0f);
	map.insert_coordinate_list(std::vector<Coord3ui>{ { 3, 3, 3 } }, 5);
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 10);
	extractor.extract(map, BitPalette::defaults(), cubes);
	VoxelRenderer renderer(ctx(), OffscreenTarget::kColourFormat);
	OffscreenTarget target(ctx(), 80, 60);
	OrbitCamera camera;
	camera.target = { 3.5f, 3.5f, 3.5f };
	camera.distance = 6.0f;
	camera.yaw_degrees = 40.0f;
	camera.pitch_degrees = 25.0f;
	const CubeBuffer* buffers[] = { &cubes };
	target.render(renderer, camera.camera(), buffers, flat_black());
	const auto image = target.read_rgba();
	EXPECT_EQ(pixel(image, 80, 40, 30), pixel_of(BitPalette::defaults().colour[5]));
	EXPECT_EQ(pixel(image, 80, 2, 2), (Pixel{ 0, 0, 0, 255 }));
	// z is up: with the pitch at 89 degrees the camera looks down and the cube is still at the centre
	camera.pitch_degrees = 89.0f;
	target.render(renderer, camera.camera(), buffers, flat_black());
	EXPECT_EQ(pixel(target.read_rgba(), 80, 40, 30), pixel_of(BitPalette::defaults().colour[5]));
}

// ---- octree ----------------------------------------------------------------------------------------------------------------------------------

namespace
{
	namespace ref = gvl_reference::octree;

	/// The cube of a node: the Morton decoded coordinates of the id on its level, 2^level voxels per side.
	Cube octree_cube(uint32_t level, uint64_t id, float side, uint32_t colour)
	{
		const ref::Voxel c = ref::unmorton(id);
		const float edge = static_cast<float>(1u << level) * side;
		return { static_cast<float>(c.x) * edge, static_cast<float>(c.y) * edge, static_cast<float>(c.z) * edge, edge, colour };
	}

	/// Gradient colour of a probabilistic occupancy, as the map test computes it.
	uint32_t gradient_colour(int value)
	{
		const float t = static_cast<float>(value + 127) / 254.0f;
		const auto channel = [](float v) { return std::clamp(v, 0.0f, 1.0f); };
		const auto byte = [&](float v) { return static_cast<uint8_t>(std::lround(channel(v) * 255.0f)); };
		return pack_colour(byte(1.5f - std::fabs(4.0f * t - 3.0f)), byte(1.5f - std::fabs(4.0f * t - 2.0f)), byte(1.5f - std::fabs(4.0f * t - 1.0f)));
	}

	/// Same position and size exactly, colours within one step per channel (the GPU rounds the gradient).
	void expect_equal_near_colour(const std::vector<Cube>& actual, const std::vector<Cube>& expected)
	{
		const auto a = sorted(actual);
		const auto e = sorted(expected);
		ASSERT_EQ(a.size(), e.size());
		for (std::size_t i = 0; i < a.size(); ++i)
		{
			EXPECT_EQ(a[i].x, e[i].x) << i;
			EXPECT_EQ(a[i].y, e[i].y) << i;
			EXPECT_EQ(a[i].z, e[i].z) << i;
			EXPECT_EQ(a[i].size, e[i].size) << i;
			EXPECT_TRUE(near(pixel_of(a[i].colour), pixel_of(e[i].colour), 1)) << i;
		}
	}

	/// Voxels with big aligned blocks (uniform nodes on levels 1, 2 and 3), scattered voxels and one in the far corner of the 2^14 grid.
	std::vector<ref::Voxel> octree_test_voxels()
	{
		std::vector<ref::Voxel> voxels;
		const auto block = [&](uint32_t x0, uint32_t y0, uint32_t z0, uint32_t n)
		{
			for (uint32_t z = 0; z < n; ++z)
				for (uint32_t y = 0; y < n; ++y)
					for (uint32_t x = 0; x < n; ++x)
						voxels.push_back({ x0 + x, y0 + y, z0 + z });
		};
		block(8, 8, 8, 8);
		block(32, 0, 16, 4);
		block(100, 70, 4, 2);
		std::mt19937_64 rng(11);
		std::uniform_int_distribution<uint32_t> any(0, 127);
		for (int i = 0; i < 300; ++i)
			voxels.push_back({ any(rng), any(rng), any(rng) });
		voxels.push_back({ 16383, 16383, 16383 });
		return voxels;
	}

	std::vector<OctreeVoxel> to_octree_voxels(const std::vector<ref::Voxel>& voxels)
	{
		std::vector<OctreeVoxel> out;
		for (const auto& v : voxels)
			out.push_back({ v.x, v.y, v.z });
		return out;
	}

	std::vector<uint64_t> morton_ids(const std::vector<ref::Voxel>& voxels, std::size_t begin, std::size_t end)
	{
		std::vector<uint64_t> ids;
		for (std::size_t i = begin; i < end && i < voxels.size(); ++i)
			ids.push_back(ref::morton(voxels[i].x, voxels[i].y, voxels[i].z));
		return ids;
	}
}

TEST_F(ViewerTest, ExtractsTheCubesOfADeterministicOctree)
{
	const auto voxels = octree_test_voxels();
	const float side = 0.25f;
	Octree tree(ctx(), OctreeKind::Deterministic, side);
	tree.build(to_octree_voxels(voxels));
	const ref::OctreeReference reference(ref::Kind::Deterministic, voxels);

	std::vector<uint8_t> occupied(256);
	for (uint32_t s = 0; s < 256; ++s)
		occupied[s] = (s & 4u) ? 1 : 0;

	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 100000);
	OctreeDisplay display;
	for (const uint32_t min_level : { 0u, 1u, 3u, 6u, 13u })
	{
		SCOPED_TRACE("min_level " + std::to_string(min_level));
		display.min_level = min_level;
		extractor.extract(tree, display, cubes);
		std::vector<Cube> expected;
		for (const auto& n : reference.extract(occupied, min_level))
			expected.push_back(octree_cube(n.level, n.id, side, display.occupied_colour));
		ASSERT_FALSE(expected.empty());
		EXPECT_EQ(cubes.found(), expected.size());
		expect_equal(cubes.download(), expected);
	}

	// on the finest level the cubes cover exactly the occupied voxels, and the aligned blocks are single big cubes
	display.min_level = 0;
	extractor.extract(tree, display, cubes);
	std::uint64_t volume = 0;
	std::set<int> edges;
	for (const Cube& c : cubes.download())
	{
		const double voxels_per_edge = static_cast<double>(c.size) / side;
		volume += static_cast<std::uint64_t>(std::llround(voxels_per_edge * voxels_per_edge * voxels_per_edge));
		edges.insert(static_cast<int>(std::lround(voxels_per_edge)));
	}
	EXPECT_EQ(volume, reference.voxel_count());
	for (const int edge : { 1, 2, 4, 8 })
		EXPECT_TRUE(edges.count(edge)) << "a cube of " << edge << " voxels per side";
}

TEST_F(ViewerTest, ExtractsTheKnownNodesOfAProbabilisticOctree)
{
	const auto voxels = octree_test_voxels();
	const float side = 0.3f;
	Octree tree(ctx(), OctreeKind::Probabilistic, side);
	tree.build(to_octree_voxels(voxels));
	// a mix of occupancies: 127 (untouched), -127 (free inserts, also new voxels), 0 (a free insert followed by an occupied one); the test does not depend
	// on which voxel ends up with which value, the expected cubes come from Octree::extract
	tree.insert_ids(morton_ids(voxels, 0, 200), OctreeDatum::Free);
	tree.insert_ids(morton_ids(voxels, 90, 200), OctreeDatum::Occupied);
	tree.insert_ids(std::vector<uint64_t>{ ref::morton(500, 500, 500), ref::morton(501, 500, 500) }, OctreeDatum::Free);
	ASSERT_NO_THROW(tree.check_tree());

	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 100000);
	for (const uint32_t min_level : { 0u, 2u })
		for (const int min_value : { 1, 0, -127, 100, 127 })
		{
			SCOPED_TRACE("min_level " + std::to_string(min_level) + ", min_value " + std::to_string(min_value));
			OctreeExtractOptions options;
			options.min_level = min_level;
			std::vector<Cube> expected;
			std::set<int> values;
			for (const OctreeCube& n : tree.extract(options))
				if (n.occupancy != -128 && n.occupancy >= min_value)
				{
					expected.push_back(octree_cube(n.level, n.id, side, gradient_colour(n.occupancy)));
					values.insert(n.occupancy);
				}
			ASSERT_FALSE(expected.empty());
			OctreeDisplay display;
			display.min_level = min_level;
			display.min_value = min_value;
			extractor.extract(tree, display, cubes);
			expect_equal_near_colour(cubes.download(), expected);
			if (min_level == 0 && min_value == -127)
				EXPECT_GE(values.size(), 3u) << "-127, 0 and 127 are all present, found: " << [&] { std::string s; for (int v : values) s += std::to_string(v) + " "; return s; }();
		}
	// a threshold below -128 still hides the unknown nodes (the fillers of the tree)
	OctreeDisplay everything;
	everything.min_value = -1000;
	extractor.extract(tree, everything, cubes);
	OctreeExtractOptions all;
	std::size_t known = 0;
	for (const OctreeCube& n : tree.extract(all))
		known += n.occupancy != -128 ? 1 : 0;
	ASSERT_GT(tree.extract(all).size(), known) << "the tree has unknown nodes";
	EXPECT_EQ(cubes.count(), known);
}

TEST_F(ViewerTest, FreeNodesOfADeterministicOctreeAreDrawnOnRequest)
{
	const float side = 0.5f;
	Octree tree(ctx(), OctreeKind::Deterministic, side);
	std::vector<OctreeVoxel> occupied;
	for (uint32_t i = 0; i < 20; ++i)
		occupied.push_back({ 40 + 3 * i, 40 + i, 40 });
	occupied.push_back({ 0, 0, 0 });
	tree.build(occupied);
	tree.insert_nodes(3, std::vector<uint64_t>{ ref::morton(10, 11, 12), ref::morton(30, 3, 3) }, OctreeDatum::Free);   // two free cubes of 8 voxels per side
	tree.insert_ids(std::vector<uint64_t>{ ref::morton(100, 100, 100), ref::morton(1, 0, 0), ref::morton(8, 8, 8) }, OctreeDatum::Free);
	ASSERT_NO_THROW(tree.check_tree());

	const uint32_t red = pack_colour(220, 30, 30), blue = pack_colour(30, 60, 200);
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 1000);
	for (const uint32_t min_level : { 0u, 1u })
	{
		SCOPED_TRACE("min_level " + std::to_string(min_level));
		OctreeExtractOptions options;
		options.min_level = min_level;
		const auto nodes = tree.extract(options);
		std::vector<Cube> both, only_occupied, only_free;
		for (const OctreeCube& n : nodes)
		{
			if (n.status & octree_format::kOccupied)
			{
				both.push_back(octree_cube(n.level, n.id, side, red));
				only_occupied.push_back(octree_cube(n.level, n.id, side, red));
			}
			else if (n.status & octree_format::kFree)
			{
				both.push_back(octree_cube(n.level, n.id, side, blue));
				only_free.push_back(octree_cube(n.level, n.id, side, blue));
			}
		}
		ASSERT_FALSE(only_occupied.empty());
		ASSERT_FALSE(only_free.empty());

		OctreeDisplay display;
		display.min_level = min_level;
		display.occupied_colour = red;
		extractor.extract(tree, display, cubes);          // free space is hidden by default
		expect_equal(cubes.download(), only_occupied);
		display.free_colour = blue;
		extractor.extract(tree, display, cubes);
		expect_equal(cubes.download(), both);
		display.occupied_colour = pack_colour(1, 2, 3, 0); // alpha 0 hides occupied
		extractor.extract(tree, display, cubes);
		expect_equal(cubes.download(), only_free);
		if (min_level == 0)
		{
			const auto free_cubes = cubes.download();
			EXPECT_EQ(std::count_if(free_cubes.begin(), free_cubes.end(), [&](const Cube& c) { return c.size == 8.0f * side; }), 2) << "the two free cubes of level 3";
		}
	}
}

TEST_F(ViewerTest, ARefinedOctreeNodeOnTheMinimumLevelStandsForItsWholeSubtree)
{
	const float side = 1.0f;
	Octree tree(ctx(), OctreeKind::Deterministic, side);
	tree.build(std::vector<OctreeVoxel>{ { 0, 0, 0 } });
	// level 1 node (0,0,0): occupied and free voxels; level 1 node (4,4,4): only a free voxel
	tree.insert_ids(std::vector<uint64_t>{ ref::morton(1, 0, 0), ref::morton(8, 8, 8) }, OctreeDatum::Free);

	const uint32_t red = pack_colour(220, 30, 30), blue = pack_colour(30, 60, 200);
	OctreeDisplay display;
	display.min_level = 1;
	display.occupied_colour = red;
	display.free_colour = blue;
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 100);
	extractor.extract(tree, display, cubes);
	// the mixed node is occupied (once, not also free), the node with only a free voxel (and unknown ones) is free
	expect_equal(cubes.download(), std::vector<Cube>{ { 0.0f, 0.0f, 0.0f, 2.0f, red }, { 8.0f, 8.0f, 8.0f, 2.0f, blue } });
	// on the finest level the free voxel of the mixed node is drawn on its own
	display.min_level = 0;
	extractor.extract(tree, display, cubes);
	expect_equal(cubes.download(), std::vector<Cube>{ { 0.0f, 0.0f, 0.0f, 1.0f, red }, { 1.0f, 0.0f, 0.0f, 1.0f, blue }, { 8.0f, 8.0f, 8.0f, 1.0f, blue } });
}

TEST_F(ViewerTest, OctreeExtractionHandlesCapacityEmptyTreesAndTheSideLength)
{
	Octree tree(ctx(), OctreeKind::Deterministic);   // no side length
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 4);
	OctreeDisplay display;
	EXPECT_THROW(extractor.extract(tree, display, cubes), VulkanError) << "neither the display nor the tree has a side length";
	display.side_length = -1.0f;
	EXPECT_THROW(extractor.extract(tree, display, cubes), VulkanError);
	display.side_length = std::nanf("");
	EXPECT_THROW(extractor.extract(tree, display, cubes), VulkanError);

	display.side_length = 0.5f;
	extractor.extract(tree, display, cubes);   // an empty tree
	EXPECT_EQ(cubes.count(), 0u);
	EXPECT_EQ(cubes.found(), 0u);

	std::vector<OctreeVoxel> far_apart;
	for (uint32_t i = 0; i < 10; ++i)
		far_apart.push_back({ 100 * i, 50 * i, 20 * i });
	tree.build(far_apart);
	extractor.extract(tree, display, cubes);
	EXPECT_EQ(cubes.found(), 10u);
	EXPECT_EQ(cubes.count(), 4u);
	EXPECT_EQ(cubes.dropped(), 6u);
	for (const Cube& c : cubes.download())
		EXPECT_EQ(c.size, 0.5f);

	// the side length of the display wins, 0 takes the one of the tree
	tree.set_voxel_side_length(2.0f);
	display.side_length = 0.0f;
	CubeBuffer large(ctx(), 100);
	extractor.extract(tree, display, large);
	for (const Cube& c : large.download())
		EXPECT_EQ(c.size, 2.0f);
	display.side_length = 0.25f;
	extractor.extract(tree, display, large);
	for (const Cube& c : large.download())
		EXPECT_EQ(c.size, 0.25f);

	display.min_level = 14;
	EXPECT_THROW(extractor.extract(tree, display, large), VulkanError);
	display.min_level = 0;
	// a second extraction replaces the content
	tree.clear();
	extractor.extract(tree, display, large);
	EXPECT_EQ(large.count(), 0u);
	EXPECT_EQ(large.found(), 0u);
}

TEST_F(ViewerTest, ABigOctreeNodeIsDrawnAsOneCube)
{
	const float side = 0.5f;
	Octree tree(ctx(), OctreeKind::Deterministic, side);
	std::vector<OctreeVoxel> block;
	for (uint32_t z = 4; z < 8; ++z)
		for (uint32_t y = 4; y < 8; ++y)
			for (uint32_t x = 4; x < 8; ++x)
				block.push_back({ x, y, z });
	tree.build(block);

	OctreeDisplay display;
	display.occupied_colour = pack_colour(200, 50, 50);
	CubeExtractor extractor(ctx());
	CubeBuffer cubes(ctx(), 10);
	extractor.extract(tree, display, cubes);
	expect_equal(cubes.download(), std::vector<Cube>{ { 2.0f, 2.0f, 2.0f, 2.0f, display.occupied_colour } });

	VoxelRenderer renderer(ctx(), OffscreenTarget::kColourFormat);
	constexpr uint32_t w = 128, h = 96;
	OffscreenTarget target(ctx(), w, h);
	ViewerCamera camera;
	camera.eye = { 3.0f, 3.0f, 10.0f };
	camera.target = { 3.0f, 3.0f, 4.0f };
	const CubeBuffer* buffers[] = { &cubes };
	target.render(renderer, camera, buffers, flat_black());
	const auto image = target.read_rgba();
	// the front face is the square (2..4) x (2..4) at z = 4
	const auto lo = project(camera, w, h, { 2.0f, 4.0f, 4.0f });
	const auto hi = project(camera, w, h, { 4.0f, 2.0f, 4.0f });
	ASSERT_LT(lo[0], hi[0]);
	ASSERT_LT(lo[1], hi[1]);
	const Pixel colour = pixel_of(display.occupied_colour);
	for (const float fx : { 0.5f, 0.1f, 0.9f })
		for (const float fy : { 0.5f, 0.1f, 0.9f })
			EXPECT_EQ(pixel(image, w, static_cast<int>(lo[0] + (hi[0] - lo[0]) * fx), static_cast<int>(lo[1] + (hi[1] - lo[1]) * fy)), colour) << fx << "," << fy;
	EXPECT_EQ(pixel(image, w, int(lo[0]) - 3, int(lo[1]) + 5), (Pixel{ 0, 0, 0, 255 }));
	EXPECT_EQ(pixel(image, w, int(hi[0]) + 3, int(hi[1]) - 5), (Pixel{ 0, 0, 0, 255 }));
	std::size_t covered = 0;
	for (uint32_t y = 0; y < h; ++y)
		for (uint32_t x = 0; x < w; ++x)
			covered += pixel(image, w, int(x), int(y)) == colour ? 1 : 0;
	EXPECT_NEAR(double(covered), double(hi[0] - lo[0]) * double(hi[1] - lo[1]), 2.0 * (double(hi[0] - lo[0]) + double(hi[1] - lo[1])));
}
